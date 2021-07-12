"""
    whoswork(blk, comm)

To which rank is block `blk` currently assigned to.

"""
function whoswork(blk, comm::MPI.Comm)
    blk % MPI.Comm_size(comm)
end

function whoswork(blk, comm::Nothing) 
    error("Communicator Nothing should not ask this.")
    return nothing
end

"""
    ismywork(blk, comm)

Returns a boolean whether the block `blk` is assigned to this rank.

"""
function ismywork(blk, comm::MPI.Comm)
    whoswork(blk, comm) == MPI.Comm_rank(comm)
end

function ismywork(blk, comm::Nothing)
    return true
end

"""
    comm_neighbors!(data, blocks, runinfo, comm)

Nearest neighbor communication where all periods t of this rank in the matrix data
will be sent to the remote ranks who own period t-1 and t+1.

This is nonblocking. An array of requests is returned.

"""
function comm_neighbors!(data::AbstractArray, blocks::OPFBlocks, runinfo::ProxALMData, comm::MPI.Comm)
	requests = MPI.Request[]
    # For each period send to t-1 and t+1
    for blk in runinfo.par_order[1,:]
        block = blocks.blkIndex[blk]
        k = block[1]
        t = block[2]
        if ismywork(blk, comm)
            for blkn in runinfo.par_order[1,:]
                blockn = blocks.blkIndex[blkn]
                kn = blockn[1]
                tn = blockn[2]
                # Neighboring period needs my work if it's not local
                if (tn == t-1 || tn == t+1) && !ismywork(blkn, comm)
                    remote = whoswork(blkn, comm)
                    push!(requests, MPI.Isend(data[:,blk], remote, t, comm))
                end
            end
        end
    end
    # For each period receive from t-1 and t+1
    for blk in runinfo.par_order[1,:]
        block = blocks.blkIndex[blk]
        k = block[1]
        t = block[2]
        if ismywork(blk, comm)
            for blkn in runinfo.par_order[1,:]
                blockn = blocks.blkIndex[blkn]
                kn = blockn[1]
                tn = blockn[2]
                # Receive neighboring period if it's not local
                if (tn == t-1 || tn == t+1) && !ismywork(blkn, comm)
                    remote = whoswork(blkn, comm)
                    buf = @view data[:,blkn]
                    push!(requests, MPI.Irecv!(buf, remote, tn, comm))
                end
            end
        end
    end
    return requests
end

function comm_neighbors!(data::AbstractArray, blocks::OPFBlocks, runinfo::ProxALMData, comm::Nothing) 
    return nothing
end

"""
    comm_wait!(requests)

Wait until the communciation requests `requests` have been fulfilled.

"""
function comm_wait!(requests::Vector{MPI.Request})
    return MPI.Waitall!(requests)
end

function comm_wait!(requests::Nothing)
    return nothing
end

"""
    comm_max(data, comm)

Collective to reduce and return the maximum over scalar `data`.

"""
function comm_max(data::Float64, comm::MPI.Comm)
    return MPI.Allreduce(data, MPI.MAX, comm)
end

function comm_max(data::Float64, comm::Nothing)
    return data
end

"""
    comm_sum(data::Float64, comm)

Collective to reduce and return the sum over scalar `data`.

"""
function comm_sum(data::Float64, comm::MPI.Comm)
    return MPI.Allreduce(data, MPI.SUM, comm)
end

function comm_sum(data::Float64, comm::Nothing)
    return data
end

"""
    comm_sum!(data, comm)

Collective to reduce the sum over array `data`.

"""
function comm_sum!(data::AbstractArray, comm::MPI.Comm)
    return MPI.Allreduce!(data, MPI.SUM, comm)
end

function comm_sum!(data::AbstractArray, comm::Nothing)
    return data
end

function comm_rank(comm::MPI.Comm)
    return MPI.Comm_rank(comm)
end

function comm_rank(comm::Nothing)
    return 0
end
    
function comm_barrier(comm::MPI.Comm)
    return MPI.Barrier(comm)
end

function comm_barrier(comm::Nothing)
    return nothing
end
