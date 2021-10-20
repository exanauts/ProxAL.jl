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
struct CommPatternT2 <: AbstractCommPattern end
struct CommPatternK2 <: AbstractCommPattern end

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
function comm_neighbors!(sdata::AbstractArray{T,2}, rdata::AbstractArray{T,2}, blocks::AbstractBlocks, runinfo::ProxALProblem, pattern::AbstractCommPattern, comm::MPI.Comm) where {T}
	requests = MPI.Request[]
    for blk in runinfo.par_order
        block = blocks.blkIndex[blk]
        k = block[1]
        t = block[2]
        if is_my_work(blk, comm)
            for blkn in runinfo.par_order
                blockn = blocks.blkIndex[blkn]
                kn = blockn[1]
                tn = blockn[2]
                if is_comm_pattern(t, tn, k, kn, pattern) && !is_my_work(blkn, comm)
                    remote = whoswork(blkn, comm)
                    if isa(pattern, CommPatternTK)
                        sbuf = @view sdata[:,blk]
                        rbuf = @view rdata[:,blkn]
                    elseif isa(pattern, CommPatternT) || isa(pattern, CommPatternT2)
                        sbuf = @view sdata[:,t]
                        rbuf = @view rdata[:,tn]
                    else
                        error("Invalid communication pattern")
                    end
                    stag = t*10+k
                    rtag = tn*10+kn
                    push!(requests, MPI.Isend(sbuf, remote, stag, comm))
                    push!(requests, MPI.Irecv!(rbuf, remote, rtag, comm))
                end
            end
        end
    end
    return requests
end


function comm_neighbors!(sdata::AbstractArray{T,3}, rdata::AbstractArray{T,3}, blocks::AbstractBlocks, runinfo::ProxALProblem, pattern::AbstractCommPattern, comm::MPI.Comm) where {T}
	requests = MPI.Request[]
    for blk in runinfo.par_order
        block = blocks.blkIndex[blk]
        k = block[1]
        t = block[2]
        if is_my_work(blk, comm)
            for blkn in runinfo.par_order
                blockn = blocks.blkIndex[blkn]
                kn = blockn[1]
                tn = blockn[2]
                if is_comm_pattern(t, tn, k, kn, pattern) && !is_my_work(blkn, comm)
                    remote = whoswork(blkn, comm)
                    if isa(pattern, CommPatternK) || isa(pattern, CommPatternK2)
                        sbuf = @view sdata[:,k,t]
                        rbuf = @view rdata[:,kn,tn]
                    else
                        error("Invalid communication pattern")
                    end
                    stag = t*10+k
                    rtag = tn*10+kn
                    push!(requests, MPI.Isend(sbuf, remote, stag, comm))
                    push!(requests, MPI.Irecv!(rbuf, remote, rtag, comm))
                end
            end
        end
    end
    return requests
end

comm_neighbors!(data, blocks, runinfo, pattern, comm) = comm_neighbors!(copy(data), data, blocks, runinfo, pattern, comm)

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
