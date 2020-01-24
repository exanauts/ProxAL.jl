include("acopf.jl")

function runAladin(opfdata::OPFData, num_partitions::Int)
    num_partitions = max(min(length(opfdata.buses), num_partitions), 1)
    network = buildNetworkPartition(opfdata, num_partitions)
    params = initializePararms(opfdata, network)

    iter = 0
    while true
         iter += 1

        # solve NLP models
        nlpmodel = solveNLP(opfdata, network, params)

        # check convergence
        (primviol, dualviol) = computeViolation(opfdata, network, params, nlpmodel)
        if primviol <= params.tol && dualviol <= params.tol
            println("converged")
            break
        end

        @printf("iter %d: primviol = %.2f, dualviol = %.2f\n", iter, primviol, dualviol)

        # solve QP
        break
    end
end



function solveNLP(opfdata::OPFData, network::OPFNetwork, params::ALADINParams)
    nlpmodel = Vector{JuMP.Model}(undef, network.num_partitions)
    for p in 1:network.num_partitions
        nlpmodel[p] = acopf_model(opfdata, network, params, p)
        nlpmodel[p], status = acopf_solve(nlpmodel[p], opfdata, network, p)
        if status != :Optimal
            error("something went wrong with status ", status)
        end

        acopf_outputAll(nlpmodel[p], opfdata, network, p)
    end

    return nlpmodel
end



function computeViolation(opfdata::OPFData, network::OPFNetwork, params::ALADINParams, nlpmodel::Vector{JuMP.Model})
    #
    # primal violation
    #
    primviol = 0.0
    for n in network.consensus_nodes
        partition = get_prop(network.graph, n, :partition)
        blocks = get_prop(network.graph, n, :blocks)
        VMtrue = getvalue(nlpmodel[partition][:Vm])[n]
        VAtrue = getvalue(nlpmodel[partition][:Va])[n]
        for p in blocks
            (p == partition) && continue
            VMcopy = getvalue(nlpmodel[p][:Vm])[n]
            VAcopy = getvalue(nlpmodel[p][:Va])[n]
            primviol += abs(VMtrue - VMcopy) + abs(VAtrue - VAcopy)
        end
    end

    #
    # dual violation
    #
    dualviol = 0.0
    for p in 1:network.num_partitions
        VM = getvalue(nlpmodel[p][:Vm])
        VA = getvalue(nlpmodel[p][:Va])
        for n in network.consensus_nodes
            (n in network.buses_bloc[p]) || continue
            dualviol += abs(VM[n] - params.VM[p][n])
            dualviol += abs(VA[n] - params.VA[p][n])
        end
    end
    dualviol *= params.ρ

    return (primviol, dualviol)
end

ARGS = ["case9"]
opfdata = opf_loaddata(ARGS[1])


