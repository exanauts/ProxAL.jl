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
    is_my_work(blk, comm)

Returns a boolean whether the block `blk` is assigned to this rank.

"""
function is_my_work(blk, comm::MPI.Comm)
    whoswork(blk, comm) == MPI.Comm_rank(comm)
end

function is_my_work(blk, comm::Nothing)
    return true
end

abstract type AbstractCommPattern end
# Symmetric communication patterns
struct CommPatternTK <: AbstractCommPattern end
struct CommPatternT <: AbstractCommPattern end
struct CommPatternK <: AbstractCommPattern end

"""
    is_comm_pattern(t, tn, k, kn, pattern)

    Do period t, tn and contingencies k, kn match the communication pattern?

"""
function is_comm_pattern(t, tn, k, kn, ::CommPatternTK)
    return (
        # Neighboring periods and base case (k == 1)
        ((tn == t-1 || tn == t+1) && kn == 1 && k == 1) ||
        # From base case to contingencies
        (tn == t && k == 1 && kn != 1) ||
        # From contingencies to base case
        (tn == t && k != 1 && kn == 1)
    )
end

function is_comm_pattern(t, tn, k, kn, ::CommPatternT)
    return (
        # Neighboring periods and base case (k == 1)
        ((tn == t-1 || tn == t+1) && kn == 1 && k == 1)
    )
end

function is_comm_pattern(t, tn, k, kn, ::CommPatternK)
    return (
        # From base case to contingencies
        (tn == t && k == 1 && kn != 1) ||
        # From contingencies to base case
        (tn == t && k != 1 && kn == 1)
    )
end

"""
    comm_neighbors!(data, blocks, runinfo, pattern, comm)

Nonblocking communication with a given pattern. An array of requests is returned.

"""
function comm_neighbors!(data::AbstractArray{T,2}, blocks::AbstractBlocks, runinfo::ProxALProblem, pattern::AbstractCommPattern, comm::MPI.Comm) where {T}
	requests = MPI.Request[]
    for blk in runinfo.blkLinIndex
        block = blocks.blkIndex[blk]
        k = block[1]
        t = block[2]
        if is_my_work(blk, comm)
            for blkn in runinfo.blkLinIndex
                blockn = blocks.blkIndex[blkn]
                kn = blockn[1]
                tn = blockn[2]
                if is_comm_pattern(t, tn, k, kn, pattern) && !is_my_work(blkn, comm)
                    remote = whoswork(blkn, comm)
                    if isa(pattern, CommPatternTK)
                        sbuf = @view data[:,blk]
                        rbuf = @view data[:,blkn]
                    elseif isa(pattern, CommPatternT)
                        sbuf = @view data[:,t]
                        rbuf = @view data[:,tn]
                    else
                        error("Invalid communication pattern")
                    end
                    push!(requests, MPI.Isend(sbuf, remote, t, comm))
                    push!(requests, MPI.Irecv!(rbuf, remote, tn, comm))
                end
            end
        end
    end
    return requests
end

function comm_neighbors!(data::AbstractArray{T,3}, blocks::AbstractBlocks, runinfo::ProxALProblem, pattern::AbstractCommPattern, comm::MPI.Comm) where {T}
	requests = MPI.Request[]
    for blk in runinfo.blkLinIndex
        block = blocks.blkIndex[blk]
        k = block[1]
        t = block[2]
        if is_my_work(blk, comm)
            for blkn in runinfo.blkLinIndex
                blockn = blocks.blkIndex[blkn]
                kn = blockn[1]
                tn = blockn[2]
                if is_comm_pattern(t, tn, k, kn, pattern) && !is_my_work(blkn, comm)
                    remote = whoswork(blkn, comm)
                    if isa(pattern, CommPatternK)
                        sbuf = @view data[:,k,t]
                        rbuf = @view data[:,kn,tn]
                    else
                        error("Invalid communication pattern")
                    end
                    push!(requests, MPI.Isend(sbuf, remote, k, comm))
                    push!(requests, MPI.Irecv!(rbuf, remote, kn, comm))
                end
            end
        end
    end
    return requests
end

function comm_neighbors!(data::AbstractArray, blocks::AbstractBlocks, runinfo::ProxALProblem, pattern::AbstractCommPattern, comm::Nothing)
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
