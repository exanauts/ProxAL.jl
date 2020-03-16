#
# Network decomposition
#

mutable struct OPFNetwork
    graph::MetaGraph
    num_partitions::Int             #number of partitions
    consensus_nodes::Vector{Int}    #nodes for which consensus must be enforced
    buses_part::Vector{Vector{Int}} #[i] := buses in partition i
    buses_bloc::Vector{Vector{Int}} #[i] := buses in partition i + their neighbors
    gener_part::Vector{Vector{Int}} #[i] := generators in partition i
    consensus_tuple::Vector{Tuple{Int, Int, Int}} # tuples (i,p,q) for which we have consensus
                                                  # VM[p][i] = VM[q][i]
                                                  # VA[p][i] = VA[q][i]
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
    consensus_tuple = Vector{Tuple{Int, Int, Int}}()

    if num_partitions <= 1
        for i in vertices(network)
            set_prop!(network, i, :partition, 1)
            set_prop!(network, i, :blocks, [1])
        end
        return OPFNetwork(network, 1, consensus_nodes, buses_part, buses_bloc, gener_part, consensus_tuple)
    end

    #
    # partition[i] = index of the partition that vertex i belongs to
    #
    partitions_metis = Metis.partition(SimpleGraph(network), num_partitions)
    partitions_unique = unique(partitions_metis)
    partition = [findfirst(x->x==partitions_metis[i], partitions_unique) for i in 1:nv(network)]
    num_partitions = length(partitions_unique)
    #partition = [1,2,1,1,1,1,2,2,2] #9-bus partition
    for i in vertices(network)
        set_prop!(network, i, :partition, partition[i])

        #
        # blocks[i] =   1. index of the partition that vertex i belongs to
        #             + 2. indices of the partitions to which its consensus neighbors belong
        set_prop!(network, i, :blocks, [partition[i]])
    end
    #nodecolor = distinguishable_colors(length(unique(partition)), #=[RGB(0.71,0.09,0.0),=# RGB(0.0,0.12,0.64))
    #GraphPlot.draw(Compose.PDF("case9_parts2.pdf", 16cm, 16cm), GraphPlot.gplot(SimpleGraph(network), layout = GraphPlot.spectral_layout, nodefillc=nodecolor[partition]))

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
    @assert num_partitions == j

    ## Build vector of consensus tuples
    for n in consensus_nodes
        partition = get_prop(network, n, :partition)
        blocks = get_prop(network, n, :blocks)
        for p in blocks
            (p == partition) && continue
            push!(consensus_tuple, (n, partition, p))
        end
    end

    return OPFNetwork(network, num_partitions, consensus_nodes, buses_part, buses_bloc, gener_part, consensus_tuple)
end
