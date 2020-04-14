#
# proximal ALM implementation
#

function runProxALM_mp(opfdata::OPFData, perturbation::Number = 0.1)
    #
    # start from perturbation of optimal solution
    #
    monolithic = get_mpmodel(opfdata)
    xstar, λstar = solve_mpmodel(monolithic, opfdata)
    zstar = computePrimalCost(xstar, opfdata)
    @printf("Optimal generation cost = %.2f\n", zstar)

    x = deepcopy(xstar); perturb(x, perturbation)
    λ = deepcopy(λstar); perturb(λ, perturbation)

    #
    # Start from scratch
    #
    #x = initializePrimalSolution(circuit, T)
    #λ = initializeDualSolution(circuit, T)


    verbose_level = 0

    plt = nothing
    if verbose_level > 1
        plt = initializePlot_iterative()
    end


    #
    # Initialize algorithmic parameters
    #
    T = size(opfdata.Pd, 2)
    maxρ = Float64(T > 1)*maximum(abs.(λstar.λ))
    params = initializeParams(maxρ; aladin = false, jacobi = true)
    params.iterlim = 10
    savedata = zeros(params.iterlim, 6)
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
        for t = 1:T
            nlpmodel[t] = get_mpmodel(opfdata, t; params = params, primal = x, dual = λ)
            t0 = time()
            nlpmodel[t], status = solve_mpmodel(nlpmodel[t], opfdata, t; initial_x = x, initial_λ = λ, params = params)
            t1 = time(); timeNLP += t1 - t0
            if status != :Optimal && status != :UserLimit
                error("something went wrong in the x-update of proximal ALM with status ", status)
            end
            #(verbose_level > 0) && acopf_outputAll(nlpmodel[p], opfdata, network, p)

            #
            # Gauss-Siedel --> immediately update
            #
            if !params.jacobi
                updatePrimalSolution(x, nlpmodel, t)
            end
        end

        #
        # Jacobi --> update the x now
        #
        if params.jacobi
            for t in 1:length(nlpmodel)
                updatePrimalSolution(x, nlpmodel, t)
            end
        end

        #
        # update the λ
        #
        updateDualSolution(λ, x, opfdata; params = params)


        #
        # Compute the primal error
        #
        primviol = computePrimalViolation(x, opfdata; lnorm = Inf)

        #
        # Compute the KKT error --> has meaningful value
        #                           only if both x and λ have been updated
        #
        dualviol = computeDualViolation(x, xprev, λ, λprev, nlpmodel, opfdata; lnorm = Inf, params = params)



        dist = computeDistance(x, xstar; lnorm = Inf)
        gencost = computePrimalCost(x, opfdata)
        gap = (gencost - zstar)/zstar

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

        savedata[iter,:] = [dist, primviol, dualviol, gap, 0, time() - tstart]
    end

    return x, λ, savedata
end


function updatePrimalSolution(x::mpPrimalSolution, nlpmodel::Vector{JuMP.Model}, t::Int)
    x.PG[t,:] = getvalue(nlpmodel[t][:Pg])
    x.QG[t,:] = getvalue(nlpmodel[t][:Qg])
    x.VM[t,:] = getvalue(nlpmodel[t][:Vm])
    x.VA[t,:] = getvalue(nlpmodel[t][:Va])
    x.SL[t,:] = getvalue(nlpmodel[t][:Sl])
end


function updateDualSolution(dual::mpDualSolution, x::mpPrimalSolution, opfdata::OPFData; params::AlgParams)
    for t=2:size(dual.λ, 1)
        for g=1:size(dual.λ, 2)
            dual.λ[t,g] += params.θ*params.ρ*(+x.PG[t-1,g] - x.PG[t,g] + x.SL[t,g] - opfdata.generators[g].ramp_agc)
        end
    end
end
