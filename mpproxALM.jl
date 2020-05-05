#
# proximal ALM implementation
#

function runProxALM_mp(opfdata::OPFData, rawdata::RawData, perturbation::Number = 0.1; options::Option = Option())
    T = options.sc_constr ? (length(rawdata.ctgs_arr) + 1) : size(opfdata.Pd, 2)


    #
    # Compute optimal solution using Ipopt
    #
    monolithic = scopf_model(opfdata, rawdata; options = options)
    t0 = time()
    xstar, λstar = solve_scmodel(monolithic, opfdata, rawdata; options = options)
    tmonolithic = time() - t0
    zstar = computePrimalCost(xstar, opfdata; options = options)
    @printf("Optimal generation cost = %.2f\n", zstar)

    #
    # start from perturbation of optimal solution
    #
    #x = deepcopy(xstar); perturb(x, perturbation)
    #λ = deepcopy(λstar); perturb(λ, perturbation)

    #
    # Start from scratch
    #
    x = initializePrimalSolution(opfdata, T; options = options)
    λ = initializeDualSolution(opfdata, T; options = options)
    if true
        nlpmodel = Vector{JuMP.Model}(undef, T)
        params = initializeParams(1.0; aladin = false, jacobi = true)
        params.τ = 0
        for t = 1:T
            params.ρ = Float64(t > 1)*ones(size(λ.λp))
            nlpmodel[t] = scopf_model(opfdata, rawdata, t; options = options, params = params, primal = x, dual = λ)
            nlpmodel[t], status = solve_scmodel(nlpmodel[t], opfdata, t; options = options, initial_x = x, initial_λ = λ, params = params)
            if status != :Optimal && status != :UserLimit
                error("something went wrong in the initialization with status ", status)
            end
            updatePrimalSolution(x, nlpmodel, t; options = options)
            if t == 1
                for s=2:T
                    nlpmodel[s] = nlpmodel[1]
                    updatePrimalSolution(x, nlpmodel, s; options = options)
                end
            end
        end
    end


    verbose_level = 0

    plt = nothing
    if verbose_level > 1
        plt = initializePlot_iterative()
    end


    #
    # Initialize algorithmic parameters
    #
    maxρ = 0.1#5.0Float64(T > 1)*max(maximum(abs.(λstar.λp)), maximum(abs.(λstar.λn)))
    params = initializeParams(maxρ; aladin = false, jacobi = true)
    params.τ = params.jacobi ? 10maxρ : 0
    params.updateρ = !(options.sc_constr && options.freq_ctrl)
    params.ρ = params.updateρ ? zeros(size(λ.λp)) : maxρ*ones(size(λ.λp))
    tol = 1e-2*ones(size(λ.λp))
    params.iterlim = 100
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
        nlpmodel = Vector{JuMP.Model}(undef, T)
        soltimes = zeros(T)
        for t = 1:T
            nlpmodel[t] = scopf_model(opfdata, rawdata, t; options = options, params = params, primal = x, dual = λ)
            t0 = time()
            nlpmodel[t], status = solve_scmodel(nlpmodel[t], opfdata, t; options = options, initial_x = x, initial_λ = λ, params = params)
            soltimes[t] = time() - t0
            if status != :Optimal && status != :UserLimit
                error("something went wrong in the x-update of proximal ALM with status ", status)
            end
            #(verbose_level > 0) && acopf_outputAll(nlpmodel[p], opfdata, network, p)

            #
            # Gauss-Siedel --> immediately update
            #
            if !params.jacobi
                updatePrimalSolution(x, nlpmodel, t; options = options)
            end
        end

        #
        # Jacobi --> update the x now
        #
        if params.jacobi
            timeNLP += maximum(soltimes)
            for t in 1:length(nlpmodel)
                updatePrimalSolution(x, nlpmodel, t; options = options)
            end
        else
            timeNLP += soltimes[1] + maximum(soltimes[2:end])
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
        dualviol, dualviolavg = computeDualViolation(x, xprev, λ, λprev, nlpmodel, opfdata; options = options, lnorm = Inf, params = params)



        dist = computeDistance(x, xstar; options = options, lnorm = Inf)
        gencost = computePrimalCost(x, opfdata; options = options)
        gap = abs((gencost - zstar)/zstar)

        if verbose_level > 1
            updatePlot_iterative(plt, iter, dist, primviol, dualviol, gap)
        end

        #
        # Various termination criteria
        converged = min(dist, max(primviol, dualviol)) <= params.tol
        if verbose_level > 0 || converged
            @printf("iter %d: primviol = %.3f, dualviol = %.3f, gap = %.3f%%, distance = %.3f\n",
                            iter, primviol, dualviol, 100gap, dist)
        end
        if false
            @printf("converged\n")
            break
        end

        savedata[iter,:] = [dist, gap, primviol, dualviol, primviolavg, dualviolavg, timeNLP, tmonolithic]
        if !isempty(options.savefile)
            writedlm(options.savefile, savedata)
        end
    end

    return x, λ, savedata
end


function updatePrimalSolution(x::mpPrimalSolution, nlpmodel::Vector{JuMP.Model}, t::Int;
                              options::Option = Option())
    x.PG[t,:] = getvalue(nlpmodel[t][:Pg])
    x.QG[t,:] = getvalue(nlpmodel[t][:Qg])
    x.VM[t,:] = getvalue(nlpmodel[t][:Vm])
    x.VA[t,:] = getvalue(nlpmodel[t][:Va])
    if options.sc_constr && options.freq_ctrl
        x.SL[t] = getvalue(nlpmodel[t][:Sl])
    else
        x.SL[t,:] = getvalue(nlpmodel[t][:Sl])
    end
end


function updateDualSolution(dual::mpDualSolution, x::mpPrimalSolution, opfdata::OPFData, tol::Array{Float64};
                            options::Option = Option(),
                            params::AlgParams)
    for t=2:size(dual.λp, 1), g=1:size(dual.λp, 2)
        viol = 0
        if options.sc_constr
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
        if !options.sc_constr || !options.freq_ctrl
            dual.λn[t,g] += params.θ*params.ρ[t,g]*errn
            dual.λp[t,g] = max(dual.λp[t,g], 0)
            dual.λn[t,g] = max(dual.λn[t,g], 0)
        end
    end
end
