#
# proximal ALM implementation
#

function runProxALM(opfdata::OPFData, rawdata::RawData, T::Int;
                    options::Option = Option(),
                    verbose_level::Int = 1,
                    parallel::Bool = false,
                    fullmodel::Bool = true,
                    maxρ::Float64 = 1.0,
                    initial_x = nothing,
                    initial_λ = nothing)
    #
    # Compute optimal solution using Ipopt
    #
    if fullmodel
        xstar, λstar, zstar, tfullmodel = solve_fullmodel(opfdata, rawdata, T; options = options)
        (verbose_level > 0) && @printf("Optimal generation cost = %.2f\n", zstar)
    else
        xstar = initializePrimalSolution(opfdata, T; options = options)
        λstar = initializeDualSolution(opfdata, T; options = options)
        zstar, tfullmodel = -1.0, -1.0
    end


    #
    # Initialize solution and the base data
    #
    x, λ, base = initializeProxALM(opfdata, rawdata, T;
                        options = options, parallel = parallel, initial_x = initial_x, initial_λ = initial_λ)


    #
    # Plot primal and dual error v/s iteration
    #
    (verbose_level > 1) && (plt = initializePlot_iterative())


    #
    # Initialize algorithmic parameters
    #
    params = initializeParams(maxρ; aladin = false, jacobi = true, options = options)
    params.ρ = params.updateρ ? zeros(size(λ.λp)) : maxρ*ones(size(λ.λp))
    tol = 1e-3*ones(size(λ.λp))
    #
    # parallelize nlp solves
    #
    serial_indices = 1:T
    parallel_indices = []
    if parallel
        optimalvector = SharedArray{Float64, 2}(base.colCount, T)
        soltimesParallel = SharedVector{Float64}(T)
        if params.jacobi || options.two_block
            serial_indices = []
            parallel_indices = 1:T
        elseif options.sc_constr
            serial_indices = 1:1
            parallel_indices = 2:T
        end
    end

    
    #
    # compute lower bound on Lyapunov sequence
    #
    Lstar = solve_fullmodel(opfdata, rawdata, T; options = options, maxρ = maxρ, compute_Lstar = true)
    (verbose_level > 0) && @printf("Lyapunov lower bound = %.2f\n", Lstar)
    lyapunovprev = Inf
    savedata = zeros(params.iterlim, 8)
    timeNLP = 0.0
    iter = 0
    while iter < params.iterlim
        iter += 1


        #
        # get a copy of the old solution --> we will use this to compute
        #                                    the KKT error at the end of the loop
        #
        xprev = deepcopy(x)
        λprev = deepcopy(λ)


        #
        # x-update step
        #
        for t in serial_indices
            t0 = time()
            opf_model_set_objective(base.nlpmodel[t], opfdata, t; options = options, params = params, primal = x, dual = λ)
            base.colValue[:,t] = opf_solve_model(base.nlpmodel[t], t)
            timeNLP += time() - t0

            #
            # Gauss-Siedel --> immediately update
            #
            if !params.jacobi && !options.two_block
                updatePrimalSolution(x, base.colValue[:,t], base.colIndex, t; options = options)
            end
        end

        if !isempty(parallel_indices)
            #
            # Parallel code
            #
            @sync for t in parallel_indices
                function xstep()
                    opfmodel = opf_model(opfdata, rawdata, t; options = options)
                    opf_model_set_objective(opfmodel, opfdata, t; options = options, params = params, primal = x, dual = λ)
                    optimalvector[:,t] = opf_solve_model(opfmodel, opfdata, t; options = options, params = params, initial_x = x, initial_λ = λ)
                end
                @async @spawn begin
                    soltimesParallel[t] = @elapsed xstep()
                end
            end

            #
            # Back to serial code
            timeNLP += maximum(soltimesParallel)
            for t in parallel_indices
                base.colValue[:,t] .= optimalvector[:,t]
                updatePrimalSolution(x, base.colValue[:,t], base.colIndex, t; options = options)
            end
        end

        #
        # Jacobi --> update the x now
        #
        if params.jacobi || options.two_block
            for t in serial_indices
                updatePrimalSolution(x, base.colValue[:,t], base.colIndex, t; options = options)
            end
        end

        #
        # Update Pg_ref
        #
        if options.sc_constr && options.two_block
            updatePrimalSolutionPgRef(x, xprev, λ, opfdata; params = params)
        end

        #
        # Update quadratic penalty slacks
        #
        if options.has_ramping && options.quadratic_penalty
            updatePrimalSolutionQuadSlacks(x, xprev, λ, opfdata; options = options, params = params)
        end

        #
        # update the λ
        #
        updateDualSolution(λ, x, opfdata, tol; options = options, params = params)
        #println("rho: avg = ", sum(params.ρ)/((T-1)*size(λ.λp, 2)), " max = ", maximum(params.ρ), " # nonzeros = ", norm(params.ρ, 0), "/", ((T-1)*size(λ.λp, 2)))


        #
        # Compute the primal error
        #
        primviol, primviolavg = computePrimalViolation(x, opfdata; options = options, lnorm = Inf)

        #
        # Compute the KKT error --> has meaningful value
        #                           only if both x and λ have been updated
        #
        dualviol, dualviolavg = computeDualViolation(x, xprev, λ, λprev, base, opfdata; options = options, lnorm = Inf, params = params)



        #
        # Compute convergence metrics
        #
        primalcost, penalty, proximal = computeLyapunovFunction(x, λ, xprev, opfdata, T; options = options, params = params)
        lyapunov = primalcost + penalty + proximal
        delta_lyapunov = lyapunovprev - lyapunov
        lyapunovprev = lyapunov
        if params.updateτ
            #if delta_lyapunov <= 0.0  && params.τ < 3.0maxρ
            if iter%8 == 0 && params.τ < 3maxρ
                params.τ += maxρ
                (verbose_level > 0) && @printf("increasing tau = %.3f (maxρ = %.3f)\n", params.τ, maxρ)
            end
        end
        dist = computeDistance(x, xstar; options = options, lnorm = Inf)
        gencost = computePrimalCost(x, opfdata; options = options)
        gap = abs((gencost - zstar)/zstar)
        (verbose_level > 1) && updatePlot_iterative(plt, iter, dist, primviol, dualviol, gap, delta_lyapunov, options.savefile)


        #
        # Termination criteria
        #
        converged = min(dist, max(primviol, dualviol)) <= params.tol
        if verbose_level > 0 || converged
            @printf("iter %d: primviol = %.3f, dualviol = %.3f, gap = %.3f%%, distance = %.3f, delta = %.3f\n",
                            iter, primviol, dualviol, 100gap, dist, delta_lyapunov)
        end

        savedata[iter,:] = [dist, gencost, primviol, dualviol, primviolavg, dualviolavg, timeNLP, tfullmodel]
        if !isempty(options.savefile)
            writedlm(options.savefile, savedata)
        end
    end

    return x, λ, savedata
end


function updatePrimalSolution(x::mpPrimalSolution, colValue::Vector{Float64}, colIndex::Dict{String, Int64}, t::Int;
                              options::Option = Option())
    for g=1:size(x.PG, 2)
        x.PG[t,g] = colValue[colIndex["Pg["*string(g)*"]"]]
    end
    for g=1:size(x.QG, 2)
        x.QG[t,g] = colValue[colIndex["Qg["*string(g)*"]"]]
    end
    for b=1:size(x.VM, 2)
        x.VM[t,b] = colValue[colIndex["Vm["*string(b)*"]"]]
    end
    for b=1:size(x.VA, 2)
        x.VA[t,b] = colValue[colIndex["Va["*string(b)*"]"]]
    end
    if options.sc_constr
        if options.freq_ctrl
            x.SL[t] = colValue[colIndex["Sl"]]
        end
        if options.two_block
            for g=1:size(x.PB, 2)
                x.PB[t,g] = colValue[colIndex["Pg_base["*string(g)*"]"]]
            end
        end
    else
        for g=1:size(x.SL, 2)
            x.SL[t,g] = colValue[colIndex["Sl["*string(g)*"]"]]
        end
    end
end

function updatePrimalSolutionPgRef(x::mpPrimalSolution, xprev::mpPrimalSolution, λ::mpDualSolution, opfdata::OPFData;
                                   params::AlgParams)
    gen = opfdata.generators
    for g=1:length(gen)
        denom = params.τ + sum(params.ρ[t,g] for t=1:size(params.ρ,1))
        if !iszero(denom)
            x.PR[g] = (1.0/denom)*(
                            (params.τ*xprev.PR[g]) +
                            sum(λ.λp[t,g] + (params.ρ[t,g]*x.PB[t,g])
                                    for t=1:size(params.ρ, 1))
                          )
            x.PR[g] = max(min(x.PR[g], gen[g].Pmax), gen[g].Pmin)
        end
    end
end

function updatePrimalSolutionQuadSlacks(x::mpPrimalSolution, xprev::mpPrimalSolution, λ::mpDualSolution, opfdata::OPFData;
                                        options::Option = Option(), params::AlgParams)
    gen = opfdata.generators
    for t=2:size(x.ZZ, 1), g=1:length(gen)
        denom = params.τ + params.ρ[t,g] + options.weight_quadratic_penalty
        if !iszero(denom)
            x.ZZ[t,g] = (1.0/denom)*(
                            (params.τ*xprev.ZZ[t,g]) - λ.λp[t,g] -
                            (params.ρ[t,g]*(x.PG[t-1,g] - x.PG[t,g] + x.SL[t,g] - gen[g].ramp_agc))
                          )
        end
    end
end

function updateDualSolution(dual::mpDualSolution, x::mpPrimalSolution, opfdata::OPFData, tol::Array{Float64, 2};
                            options::Option = Option(), params::AlgParams)
    for t=1:size(dual.λp, 1), g=1:size(dual.λp, 2)
        if t == 1 && !(options.sc_constr && options.two_block)
            continue
        end
        viol = 0
        if options.sc_constr && options.two_block
            errp = x.PB[t,g] - x.PR[g]
            errn = 0
            viol = abs(errp)
        elseif options.sc_constr
            if options.freq_ctrl
                errp = +x.PG[1,g] - x.PG[t,g] + (opfdata.generators[g].alpha*x.SL[t])
                errn = 0
                viol = abs(errp)
            else
                errp = +x.PG[1,g] - x.PG[t,g] - opfdata.generators[g].scen_agc
                errn = -x.PG[1,g] + x.PG[t,g] - opfdata.generators[g].scen_agc
                viol = max(errp, errn, 0)
            end
        elseif options.has_ramping && options.quadratic_penalty
            errp = +x.PG[t-1,g] - x.PG[t,g] + x.SL[t,g] + x.ZZ[t,g] - opfdata.generators[g].ramp_agc
            errn = 0
            viol = abs(errp)
        elseif options.has_ramping
            errp = +x.PG[t-1,g] - x.PG[t,g] - opfdata.generators[g].ramp_agc
            errn = -x.PG[t-1,g] + x.PG[t,g] - opfdata.generators[g].ramp_agc
            viol = max(errp, errn, 0)
        end
        if params.updateρ
            if params.ρ[t,g] < params.maxρ && viol > tol[t,g]
                params.ρ[t,g] = min(params.ρ[t,g] + 0.1params.maxρ, params.maxρ)
            else
                if viol <= tol[t,g]
                    tol[t,g] = max(tol[t,g]/1.2, params.zero)
                end
            end
        end
        dual.λp[t,g] += params.θ*params.ρ[t,g]*errp
        if options.sc_constr && (options.freq_ctrl || options.two_block)
            continue
        end
        if options.has_ramping && options.quadratic_penalty
            continue
        end
        dual.λn[t,g] += params.θ*params.ρ[t,g]*errn
        dual.λp[t,g] = max(dual.λp[t,g], 0)
        dual.λn[t,g] = max(dual.λn[t,g], 0)
    end
end

function solve_fullmodel(opfdata::OPFData, rawdata::RawData, T::Int; options::Option = Option(), maxρ::Float64 = 1.0, compute_Lstar::Bool = false)
    if compute_Lstar
        opfmodel = opf_fullmodel(opfdata, rawdata, T; options = options, maxρ = maxρ, compute_Lstar = true)
        optimize!(opfmodel)
        status = termination_status(opfmodel)
        if  status != MOI.OPTIMAL &&
            status != MOI.ALMOST_OPTIMAL &&
            status != MOI.LOCALLY_SOLVED &&
            status != MOI.ALMOST_LOCALLY_SOLVED
            println("Quadratic penalty model status is not optimal: ", status)
            return 0
        end
        return objective_value(opfmodel)
    end
    opfmodel = opf_fullmodel(opfdata, rawdata, T; options = options)
    xstar, λstar = opf_solve_fullmodel(opfmodel, opfdata, rawdata, T; options = options)
    objvalue = computePrimalCost(xstar, opfdata; options = options)
    solvetime = JuMP.solve_time(opfmodel)
    return xstar, λstar, objvalue, solvetime
end

function initializeProxALM(opfdata::OPFData, rawdata::RawData, T::Int;
                           options::Option = Option(),
                           parallel::Bool = false,
                           initial_x = nothing,
                           initial_λ = nothing)
    #
    # initialize solution vectors
    #
    if initial_x == nothing
        x = initializePrimalSolution(opfdata, T; options = options)
        sequential_solve = true
    else
        x = deepcopy(initial_x)
        sequential_solve = false
    end
    if initial_λ == nothing
        λ = initializeDualSolution(opfdata, T; options = options)
    else
        λ = deepcopy(initial_λ)
    end

    
    #
    # NLP subproblem data
    #
    nlpmodel = Vector{JuMP.Model}(undef, T)
    colCount = 0
    colIndex = Dict{String, Int64}()
    colLower = zeros(colCount, T)
    colUpper = zeros(colCount, T)
    colValue = zeros(colCount, T)


    #
    # Make an initial sequential solve
    #
    params = initializeParams(1.0; aladin = false, jacobi = true)
    params.τ = 0
    for t = 1:T
        #
        # Scenario-1/Period-1 model is standard ACOPF
        # Scenario-t/Period-t model has a quadratic penalty term from the coupling constraints
        #
        params.ρ = Float64(t > 1)*ones(size(λ.λp))
        nlpmodel[t] = opf_model(opfdata, rawdata, t; options = options)

        #
        # Initialize the column values
        #
        if t == 1
            colCount = JuMP.num_variables(nlpmodel[t])
            colLower = zeros(colCount, T)
            colUpper = zeros(colCount, T)
            colValue = zeros(colCount, T)
        end
        @assert colCount == JuMP.num_variables(nlpmodel[t])

        #
        # solve the subproblem
        #
        opf_model_set_objective(nlpmodel[t], opfdata, t; options = options, params = params, primal = x, dual = λ)

        #
        # Solve only t=1
        # We will solve t=2:T after the loop in parallel
        #
        if t == 1 || (!parallel && sequential_solve)
            colValue[:,t] = opf_solve_model(nlpmodel[t], t)
        end


        #
        # Update the column values
        #
        all_vars = JuMP.all_variables(nlpmodel[t])
        for var in all_vars
            if t == 1
                j = JuMP.optimizer_index(var).value
                colIndex[JuMP.name(var)] = j
            else
                j = colIndex[JuMP.name(var)]
            end
            colLower[j, t] = has_lower_bound(var) ? lower_bound(var) : -Inf
            colUpper[j, t] = has_upper_bound(var) ? upper_bound(var) : +Inf
        end


        #
        # Update the primal solution data structure
        #
        if sequential_solve
            if t == 1 || !parallel
                updatePrimalSolution(x, colValue[:,t], colIndex, t; options = options)
            end
        end
        if t == 1
            #=
            for s=2:T
                updatePrimalSolution(x, colValue[:,t], colIndex, s; options = options)
            end
            =#
            # update Pg_ref
            if options.sc_constr && options.two_block
                x.PR .= x.PG[1,:]
            end
        end
        if t == T
            # update ZZ
            if options.has_ramping && options.quadratic_penalty
                for s = 2:T, g=1:size(x.ZZ, 2)
                    x.ZZ[s,g] = -(+x.PG[s-1,g] - x.PG[s,g] + x.SL[s,g] - opfdata.generators[g].ramp_agc)
                end
            end
        end
    end

    if parallel && sequential_solve
        optimalvector = SharedArray{Float64, 2}(colCount, T)
        params.ρ = 1e-3ones(size(λ.λp))
        @sync for t=2:T
            function xstep()
                opfmodel = opf_model(opfdata, rawdata, t; options = options)
                opf_model_set_objective(opfmodel, opfdata, t; options = options, params = params, primal = x, dual = λ)
                optimalvector[:,t] = opf_solve_model(opfmodel, t)
            end
            @async @spawn begin
                xstep()
            end
        end
        for t=2:T
            colValue[:,t] .= optimalvector[:,t]
            updatePrimalSolution(x, colValue[:,t], colIndex, t; options = options)
        end
    end

    return x, λ, AlgNLPData(nlpmodel, colCount, colIndex, colLower, colUpper, colValue)
end
