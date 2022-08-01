struct LocalStorage{VT,N,M,N1} <: AbstractArray{VT,N1}
    data::Array{Union{Vector{VT}, VT, Nothing},N}
    n::Int
    entries::Int64
    K::Int64
    T::Int64
end

LocalStorage{2,M}(args...) where {M} = LocalStorage{Float64,2,M}(args...)
LocalStorage{2,1}(args...) = LocalStorage{Float64,2,1}(args...)
LocalStorage{1,M}(args...) where {M} = LocalStorage{Float64,1,M}(args...)

function LocalStorage{VT,2,M}(K::Int, T::Int, blocks::OPFBlocks, blockindices::Tuple{Vector{Int64},Vector{Int64}}) where {VT,M}
    arrays = Array{Union{Vector{VT},Nothing},2}(undef, K, T)
    tmp = zeros(VT, M)
    entries = 0
    for i in 1:length(arrays)
        arrays[i] = nothing
    end
    for blk in vcat(blockindices[1], blockindices[2])
        block = blocks.blkIndex[blk]
        k = block[1]
        t = block[2]
        arrays[k,t] = Vector{VT}(tmp)
        entries += 1
    end
    LocalStorage{VT,2,M,3}(arrays, M, entries, K, T)
end

function LocalStorage{VT,1,M}(T::Int, blocks::OPFBlocks, blockindices::Tuple{Vector{Int64},Vector{Int64}}) where {VT,M}
    arrays = Array{Union{Vector{VT},Nothing},1}(undef, T)
    tmp = zeros(VT, M)
    entries = 0
    for i in 1:length(arrays)
        arrays[i] = nothing
    end
    for blk in vcat(blockindices[1], blockindices[2])
        block = blocks.blkIndex[blk]
        k = block[1]
        t = block[2]
        arrays[t] = Vector{VT}(tmp)
        entries += 1
    end
    LocalStorage{VT,1,M,2}(arrays, M, entries, 0, T)
end

function LocalStorage{VT,2,1}(K::Int, T::Int, blocks::OPFBlocks, blockindices::Tuple{Vector{Int64},Vector{Int64}}) where {VT}
    arrays = Array{Union{VT, Nothing},2}(undef, K, T)
    entries = 0
    for i in 1:length(arrays)
        arrays[i] = nothing
    end
    for blk in vcat(blockindices[1], blockindices[2])
        block = blocks.blkIndex[blk]
        k = block[1]
        t = block[2]
        arrays[k,t] = zero(VT)
        entries += 1
    end
    LocalStorage{VT,2,1,2}(arrays, 1, entries, K, T)
end

Base.size(L::LocalStorage{VT,2,M,3}) where {VT,M} = (L.n, L.K, L.T)
Base.size(L::LocalStorage{VT,1,M,2}) where {VT,M} = (L.n, L.T)
Base.size(L::LocalStorage{VT,2,1,2}) where {VT} = (L.K, L.T)
Base.size(L::LocalStorage, i::Int64) = (L.n, L.K, L.T)[i]

function Base.getindex(L::LocalStorage{VT,2,M,3}, ::Colon, I1::UnitRange{Int64}, I2::Int64) where {VT,M}
    m = Matrix{VT}(undef, L.n, length(I1))
    for (i, i1) in enumerate(I1)
        m[:,i] = L.data[i1,I2][:]
    end
    return m
end

function Base.getindex(L::LocalStorage{VT,2,1,2}, I1::UnitRange{Int64}, I2::Int64) where {VT}
    v = Vector{VT}(undef, length(I1))
    for (i, i1) in enumerate(I1)
        v[i] = L.data[i1,I2]
    end
    return v
end

function Base.getindex(L::LocalStorage{VT,2,1,2}, I1::Int64, I2::Int64) where {VT}
    return L.data[I1,I2]
end

function Base.getindex(L::LocalStorage{VT,1,M,2}, ::Colon, I1::Int64) where {VT,M}
    v = Vector{VT}(L.data[I1][:])
    # @show pointer(L.data[I1])
    # @show typeof(L.data[I1])
    # v = Vector{VT}(undef, L.n)
    # v .= L.data[I1][:]
    return v
end

function Base.getindex(L::LocalStorage{VT,2,M,3}, ::Colon, I1::Int64, I2::Int64) where {VT,M}
    m = view(L.data[I1,I2],:)
    return m
end

function Base.getindex(L::LocalStorage{VT,2,M,3}, I1::Int64, I2::Colon, I3::Colon) where {VT,M}
    m = getindex(L.data,:,:)
    ret = getindex.(m,I1)
    return ret
end

function Base.getindex(L::LocalStorage{VT,1,M,2}, I1::Int64, I2::Int64) where {VT,M}
    m = getindex(L.data,I2)
    ret = getindex(m,I1)
    return ret
end

function Base.getindex(L::LocalStorage{VT,2,M,3}, I1::Int64, I2::Int64, I3::Int64) where {VT,M}
    m = getindex(L.data,I2,I3)
    ret = getindex(m,I1)
    return ret
end

function Base.setindex!(L::LocalStorage{VT,1,M,2}, x::Vector{Float64}, ::Colon, I1::Int64) where {VT,M}
    L.data[I1][:] = x[:]
end

function Base.setindex!(L::LocalStorage{VT,1,M,2}, x::Float64, I1::Int64, I2::Int64) where {VT,M}
    L.data[I2][I1] = x
end

function Base.setindex!(L::LocalStorage{VT,2,M,3}, x::Array{Float64,1}, ::Colon, I1::Int64, I2::Int64) where {VT,M}
    L.data[I1,I2][:] = x
end

function Base.setindex!(L::LocalStorage{VT,2,M,3}, x::Float64, ::Colon, I1, I2) where {VT,M}
    L.data[I1,I2] .= x
end

function Base.setindex!(L::LocalStorage{VT,2,M,3}, x::Float64, I1::Int64, I2::Int64, I3::Int64) where {VT,M}
    L.data[I2,I3][I1] = x
end

function Base.setindex!(L::LocalStorage{VT,2,1,2}, x::Float64, I1::Int64, I2::Int64) where {VT}
    L.data[I1,I2] = x
end

function Base.setindex!(L::LocalStorage{VT,2,M,3}, x::Vector{Float64}, ::Colon, I1::UnitRange{Int64}, I2::Int64) where {VT,M}
    from = 1
    to = 0
    for i in I1
        to = to + length(L.data[i,I2][:])
        L.data[i,I2][:] = x[from:to]
        from = to + 1
    end
end
