#
# proximal ALM implementation
#

function runProxALM(opfdata::OPFData, rawdata::RawData, T::Int; options::Option = Option(), verbose_level::Int = 1)
    #
    # Compute optimal solution using Ipopt
    #
    xstar, λstar, zstar, tfullmodel = solve_fullmodel(opfdata, rawdata, T; options = options)
    (verbose_level > 0) && @printf("Optimal generation cost = %.2f\n", zstar)

    #
    # Initialize solution and the basemodel
    #
    x, λ, base = initializeProxALM(opfdata, rawdata, T; options = options)


    #
    # Plot primal and dual error v/s iteration
    #
    (verbose_level > 1) && (plt = initializePlot_iterative())


    #
    # Initialize algorithmic parameters
    #
    maxρ = 0.5#1.0Float64(T > 1)*max(maximum(abs.(λstar.λp)), maximum(abs.(λstar.λn)))
    params = initializeParams(maxρ; aladin = false, jacobi = true, options = options)
    params.ρ = params.updateρ ? zeros(size(λ.λp)) : maxρ*ones(size(λ.λp))
    tol = 1e-2*ones(size(λ.λp))
    savedata = zeros(params.iterlim, 8)
    tstart = time()
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
        soltimes = zeros(T)
        for t = 1:T
            t0 = time()
            opf_model_set_objective(base.nlpmodel[t], opfdata, t; options = options, params = params, primal = x, dual = λ)
            base.colValue[:,t] = opf_solve_model(base.nlpmodel[t])
            soltimes[t] = time() - t0

            #
            # Gauss-Siedel --> immediately update
            #
            if !params.jacobi && !options.two_block
                updatePrimalSolution(x, base.colValue[:,t], base.colIndex, t; options = options)
            end
        end

        #
        # Jacobi --> update the x now
        #
        if params.jacobi
            timeNLP += maximum(soltimes)
            for t in 1:T
                updatePrimalSolution(x, base.colValue[:,t], base.colIndex, t; options = options)
            end
        else
            timeNLP += soltimes[1] + maximum(soltimes[2:end])
        end

        #
        # Update Pg_ref
        #
        if options.sc_constr && options.two_block
            updatePrimalSolutionPgRef(x, xprev, λ, opfdata; params = params)
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
        dist = computeDistance(x, xstar; options = options, lnorm = Inf)
        gencost = computePrimalCost(x, opfdata; options = options)
        gap = abs((gencost - zstar)/zstar)
        (verbose_level > 1) && updatePlot_iterative(plt, iter, dist, primviol, dualviol, gap)


        #
        # Termination criteria
        #
        converged = min(dist, max(primviol, dualviol)) <= params.tol
        if verbose_level > 0 || converged
            @printf("iter %d: primviol = %.3f, dualviol = %.3f, gap = %.3f%%, distance = %.3f\n",
                            iter, primviol, dualviol, 100gap, dist)
        end

        savedata[iter,:] = [dist, gap, primviol, dualviol, primviolavg, dualviolavg, timeNLP, tfullmodel]
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
        if !options.sc_constr || !options.freq_ctrl && !options.two_block
            dual.λn[t,g] += params.θ*params.ρ[t,g]*errn
            dual.λp[t,g] = max(dual.λp[t,g], 0)
            dual.λn[t,g] = max(dual.λn[t,g], 0)
        end
    end
end

function solve_fullmodel(opfdata::OPFData, rawdata::RawData, T::Int; options::Option = Option())
    opfmodel = opf_fullmodel(opfdata, rawdata, T; options = options)
    xstar, λstar = opf_solve_fullmodel(opfmodel, opfdata, rawdata, T; options = options)
    objvalue = computePrimalCost(xstar, opfdata; options = options)
    solvetime = JuMP.solve_time(opfmodel)
    return xstar, λstar, objvalue, solvetime
end

function initializeProxALM(opfdata::OPFData, rawdata::RawData, T::Int; options::Option = Option())
    #
    # initialize solution vectors
    #
    x = initializePrimalSolution(opfdata, T; options = options)
    λ = initializeDualSolution(opfdata, T; options = options)

    
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
        colValue[:,t] = opf_solve_model(nlpmodel[t])


        #
        # Update the column values
        #
        all_vars = JuMP.all_variables(nlpmodel[t])
        for var in all_vars
            j = JuMP.optimizer_index(var).value
            colLower[j, t] = has_lower_bound(var) ? lower_bound(var) : -Inf
            colUpper[j, t] = has_upper_bound(var) ? upper_bound(var) : +Inf
            if t == 1
                colIndex[JuMP.name(var)] = j
            end
            @assert colIndex[JuMP.name(var)] == j
        end


        #
        # Update the primal solution data structure
        #
        updatePrimalSolution(x, colValue[:,t], colIndex, t; options = options)
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
    end

    return x, λ, AlgNLPData(nlpmodel, colCount, colIndex, colLower, colUpper, colValue)
end
