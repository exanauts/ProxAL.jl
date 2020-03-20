#
# proximal ALM implementation
#

function runProxALM(case::String, num_partitions::Int, perturbation::Number = 0.1)
    opfdata = opf_loaddata(case)
    num_partitions = max(min(length(opfdata.buses), num_partitions), 1)
    network = buildNetworkPartition(opfdata, num_partitions)


    #
    # start from perturbation of optimal solution
    #
    monolithic = acopf_model_monolithic(opfdata, network)
    xstar, λstar = acopf_solve_monolithic(monolithic, opfdata, network)
    @printf("Optimal generation cost = %.2f\n", computePrimalCost(xstar, opfdata))
    
    x = deepcopy(xstar); perturb(x, perturbation)
    λ = deepcopy(λstar); perturb(λ, perturbation)

    #
    # Start from scratch
    #
    #x = initializePrimalSolution(opfdata, network)
    #λ = initializeDualSolution(opfdata, network)


    verbose_level = 0
    
    plt = nothing
    if verbose_level > 1
        plt = initializePlot_iterative()
    end


    #
    # Initialize algorithmic parameters
    #
    if num_partitions > 1
        ρVM = maximum([abs(λstar.λVM[key]) for key in keys(λstar.λVM)])
        ρVA = maximum([abs(λstar.λVA[key]) for key in keys(λstar.λVA)])
        maxρ = 5.0max(ρVM, ρVA)
        #maxρ = num_partitions + 2.0 #case30
    else
        maxρ = 1.0
    end
    params = initializeParams(maxρ; aladin = false, jacobi = true)
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
        nlpmodel = Vector{JuMP.Model}(undef, network.num_partitions)
        for p in 1:network.num_partitions
            nlpmodel[p] = acopf_model(opfdata, network, p; params = params, primal = x, dual = λ)
            t0 = time()
            nlpmodel[p], status = acopf_solve(nlpmodel[p], opfdata, network, p; initial = x)
            t1 = time(); timeNLP += t1 - t0
            if status != :Optimal && status != :UserLimit
                error("something went wrong in the x-update of proximal ALM with status ", status)
            end
            #(verbose_level > 0) && acopf_outputAll(nlpmodel[p], opfdata, network, p)

            #
            # Gauss-Siedel --> immediately update
            #
            if !params.jacobi
                updatePrimalSolution(x, nlpmodel, network, p)
            end
        end

        #
        # Jacobi --> update the x now
        #
        if params.jacobi
            for p in 1:length(nlpmodel)
                updatePrimalSolution(x, nlpmodel, network, p)
            end
        end

        #
        # update the λ
        #
        updateDualSolution(λ, x; params = params)


        #
        # Compute the primal error
        #
        primviol = computePrimalViolation(x, network; lnorm = Inf)

        #
        # Compute the KKT error --> has meaningful value
        #                           only if both x and λ have been updated
        #
        dualviol = computeDualViolation(x, xprev, nlpmodel, network; lnorm = Inf, params = params)

        #
        # Alternate metric of infeasibility
        #
        primfeas, kkterror = computePrimalDualError_manual(opfdata, network, nlpmodel, x; lnorm = Inf, compute_dual_error = true)
        


        #
        # Update the parameters (optional)
        #
        x_distprev = computeDistance(x, xprev; lnorm = 2)
        λ_distprev = computeDistance(λ, λprev; lnorm = 2)
        prox_value = x_distprev^2 + λ_distprev^2
        #
        # compute the scaled distance between old (x, λ) and new (x, λ)
        # See Deng, Lai, Peng and Yin, J. Sci. Comput., 2016
        #
        delta = 0.5params.τ*x_distprev^2
        for p in 1:network.num_partitions
            for b in network.buses_bloc[p]
                if b in network.consensus_nodes
                    delta += params.ρ*(x.VM[p][b] - xprev.VM[p][b])^2
                    delta += params.ρ*(x.VA[p][b] - xprev.VA[p][b])^2
                end
            end
        end
        delta += (2.0 - params.θ)/(params.θ^2*params.ρ)*λ_distprev^2
        for (b, p, q) in network.consensus_tuple
            key = (b, p, q)
            delta -= (2.0/params.θ)*(λ.λVM[key]-λprev.λVM[key])*((x.VM[p][b] - xprev.VM[p][b]) - (x.VM[q][b] - xprev.VM[q][b]))
            delta -= (2.0/params.θ)*(λ.λVA[key]-λprev.λVA[key])*((x.VA[p][b] - xprev.VA[p][b]) - (x.VA[q][b] - xprev.VA[q][b]))
        end



        #
        # ρ update (based on Boyd's monograph)
        #
        if params.updateρ
            if (primviol > 10dualviol)
                params.ρ *= 2.0
            elseif (dualviol > 10primviol)
                params.ρ /= 2.0
            end
        end
        #
        # τ update (based on Deng, Lai, Peng and Yin, J. Sci. Comput., 2016)
        #
        if params.updateτ
            if (delta < 0.01(x_distprev^2+λ_distprev^2))
                params.τ *= 2.0
            end
        end
        
        dist = computeDistance(x, xstar; lnorm = Inf)
        gencost = computePrimalCost(x, opfdata)

        if verbose_level > 1
            updatePlot_iterative(plt, iter, dist, primviol, dualviol, primfeas, kkterror)
        end

        #
        # Various termination criteria
        converged = min(dist, max(primfeas, dualviol)) <= params.tol
        #converged = min(dist, max(primviol, min(dualviol, kkterror))) <= params.tol


        if verbose_level > 0 || converged
            @printf("iter %d: primviol = %.2f, primfeas = %.2f, dualviol = %.2f, gencost = %.2f, primdist = %.3f\n", iter, primviol, primfeas, dualviol, gencost, dist)
        end
        if false
            @printf("converged\n")
            break
        end

        savedata[iter,:] = [dist, primviol, dualviol, primfeas, kkterror, timeNLP, 0, time() - tstart]
    end

    writedlm(getDataFilename("", case, "proxALM", num_partitions, perturbation, params.jacobi), savedata)


    return x, λ
end


function updatePrimalSolution(x::PrimalSolution, nlpmodel::Vector{JuMP.Model}, network::OPFNetwork, partition_idx::Int)
    p = partition_idx
    for b in network.buses_bloc[p]
        x.VM[p][b] = getvalue(nlpmodel[p][:Vm][b])
        x.VA[p][b] = getvalue(nlpmodel[p][:Va][b])
    end
    for g in network.gener_part[p]
        x.PG[p][g] = getvalue(nlpmodel[p][:Pg][g])
        x.QG[p][g] = getvalue(nlpmodel[p][:Qg][g])
    end
end


function updateDualSolution(dual::DualSolution, x::PrimalSolution; params::AlgParams)
    for (j, p, q) in keys(dual.λVM)
        dual.λVM[(j, p, q)] += params.θ*params.ρ*(x.VM[p][j] - x.VM[q][j])
        dual.λVA[(j, p, q)] += params.θ*params.ρ*(x.VA[p][j] - x.VA[q][j])
    end
end


