using LightGraphs, MetaGraphs, Metis

mutable struct OPFNetwork
    graph::MetaGraph
    num_partitions::Int             #number of partitions
    consensus_nodes::Vector{Int}    #nodes for which consensus must be enforced
    buses_part::Vector{Vector{Int}} #[i] := buses in partition i
    buses_bloc::Vector{Vector{Int}} #[i] := buses in partition i + their neighbors
    gener_part::Vector{Vector{Int}} #[i] := generators in partition i
end


# Builds the graph G
function buildGraph(buses, lines, busDict)
    nbus = length(buses)
    G = MetaGraph(nbus)
    for i in 1:length(lines)
        add_edge!(G, busDict[lines[i].from], busDict[lines[i].to])
    end

    return G
end

# Builds the network partition data
function buildNetworkPartition(opfdata, num_partitions::Int)
    network = buildGraph(opfdata.buses, opfdata.lines, opfdata.BusIdx)
    consensus_nodes = Vector{Int}()
    buses_part = [1:length(opfdata.buses)]
    buses_bloc = [1:length(opfdata.buses)]
    gener_part = [1:length(opfdata.generators)]

    if num_partitions <= 1
        for i in vertices(network)
            set_prop!(network, i, :partition, 1)
            set_prop!(network, i, :blocks, [1])
        end
        return OPFNetwork(network, 1, consensus_nodes, buses_part, buses_bloc, gener_part)
    end

    #
    # partition[i] = index of the partition that vertex i belongs to
    #
    partition = Metis.partition(SimpleGraph(network), num_partitions)
    for i in vertices(network)
        set_prop!(network, i, :partition, partition[i])

        #
        # blocks[i] =   1. index of the partition that vertex i belongs to
        #             + 2. indices of the partitions to which its consensus neighbors belong
        set_prop!(network, i, :blocks, [partition[i]])
    end

    idx = 1
    for e in edges(network)
        if partition[src(e)] != partition[dst(e)]
            push!(consensus_nodes, src(e))
            push!(consensus_nodes, dst(e))
            set_prop!(network, e, :index, idx); idx += 1
            
            v = get_prop(network, src(e), :blocks); push!(v, partition[dst(e)])
            set_prop!(network, src(e), :blocks, unique(v))

            v = get_prop(network, dst(e), :blocks); push!(v, partition[src(e)])
            set_prop!(network, dst(e), :blocks, unique(v))
        end
    end
    sort!(consensus_nodes)
    unique!(consensus_nodes)
    

    buses_part = Vector{Vector{Int}}()
    buses_bloc = Vector{Vector{Int}}()
    gener_part = Vector{Vector{Int}}()
    j = 0
    for idx in 1:num_partitions
        isempty(collect(filter_vertices(network, (G,v)->get_prop(G, v, :partition) == idx))) && continue
        j += 1
        push!(buses_part, collect(filter_vertices(network, (G,v)->get_prop(G, v, :partition) == idx)))
        push!(buses_bloc, collect(filter_vertices(network, (G,v)->idx in get_prop(G, v, :blocks))))
        push!(gener_part, Vector{Int}())
        for i in buses_part[j]
            for g in opfdata.BusGenerators[i]
                push!(gener_part[j], g)
            end
        end
    end
    num_partitions = j

    return OPFNetwork(network, num_partitions, consensus_nodes, buses_part, buses_bloc, gener_part)
end
