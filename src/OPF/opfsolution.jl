#
# Data structures to represent primal and dual solutions
#
mutable struct OPFPrimalSolution <: AbstractPrimalSolution
    Pg
    Qg
    Vm
    Va
    ωt
    St
    Zt
    Sk
    Zk
    sigma_real
    sigma_imag
    sigma_lineFr
    sigma_lineTo

    function OPFPrimalSolution(
        opfdata::OPFData,
        modelinfo::ModelInfo,
        blocks::Union{AbstractBlocks,Nothing} = nothing,
        blkLocalIndices::Nothing = nothing,
        blkLinkedIndices::Nothing = nothing,
    )
        buses = opfdata.buses
        gens  = opfdata.generators
        ngen  = length(gens)
        nbus  = length(buses)
        nline = length(opfdata.lines)
        T = modelinfo.num_time_periods
        K = modelinfo.num_ctgs + 1 # base case counted separately
        @assert T >= 1
        @assert K >= 1

        Pg = zeros(ngen,K,T)
        Qg = zeros(ngen,K,T)
        Vm = zeros(nbus,K,T)
        Va = zeros(nbus,K,T)
        ωt = zeros(K,T)
        St = zeros(ngen,T)
        Zt = zeros(ngen,T)
        Sk = zeros(ngen,K,T)
        Zk = zeros(ngen,K,T)
        if modelinfo.allow_constr_infeas
            sigma_real = zeros(nbus,K,T)
            sigma_imag = zeros(nbus,K,T)
            sigma_lineFr = zeros(nline,K,T)
            sigma_lineTo = zeros(nline,K,T)
        else
            sigma_real = 0
            sigma_imag = 0
            sigma_lineFr = 0
            sigma_lineTo = 0
        end

        for i=1:ngen
            Pg[i,:,:] .= 0.5*(gens[i].Pmax + gens[i].Pmin)
            Qg[i,:,:] .= 0.5*(gens[i].Qmax + gens[i].Qmin)
        end
        for i=1:nbus
            Vm[i,:,:] .= 0.5*(buses[i].Vmax + buses[i].Vmin)
            Va[i,:,:] .= opfdata.buses[opfdata.bus_ref].Va
        end
        if T > 1
            for i=1:ngen
                St[i,2:T] .= gens[i].ramp_agc
            end
        end
        if K > 1
            for i=1:ngen
                Sk[i,2:K,:] .= gens[i].scen_agc
            end
        end

        new(Pg,Qg,Vm,Va,ωt,St,Zt,Sk,Zk,sigma_real,sigma_imag,sigma_lineFr,sigma_lineTo)
    end

    function OPFPrimalSolution(
        opfdata::OPFData,
        modelinfo::ModelInfo,
        blocks::AbstractBlocks,
        blkLocalIndices::Array{Int64},
        blkLinkedIndices::Array{Int64},
    )
        buses = opfdata.buses
        gens  = opfdata.generators
        ngen  = length(gens)
        nbus  = length(buses)
        nline = length(opfdata.lines)
        T = modelinfo.num_time_periods
        K = modelinfo.num_ctgs + 1 # base case counted separately
        @assert T >= 1
        @assert K >= 1

        blockindices = (blkLocalIndices, blkLinkedIndices)
        Pg = LocalStorage{2,ngen}(K, T, blocks, blockindices)
        Qg = LocalStorage{2,ngen}(K, T, blocks, blockindices)
        Vm = LocalStorage{2,nbus}(K, T, blocks, blockindices)
        Va = LocalStorage{2,nbus}(K, T, blocks, blockindices)
        ωt = LocalStorage{2,1}(K, T, blocks, blockindices)
        St = LocalStorage{1,ngen}(T, blocks, blockindices)
        Zt = LocalStorage{1,ngen}(T, blocks, blockindices)
        Sk = LocalStorage{2,ngen}(K, T, blocks, blockindices)
        Zk = LocalStorage{2,ngen}(K, T, blocks, blockindices)
        if modelinfo.allow_constr_infeas
            sigma_real = LocalStorage{2,nbus}(K, T, blocks, blockindices)
            sigma_imag = LocalStorage{2,nbus}(K, T, blocks, blockindices)
            sigma_lineFr = LocalStorage{2,nline}(K, T, blocks, blockindices)
            sigma_lineTo = LocalStorage{2,nline}(K, T, blocks, blockindices)
        else
            sigma_real = 0
            sigma_imag = 0
            sigma_lineFr = 0
            sigma_lineTo = 0
        end
        for blk in vcat(blockindices[1], blockindices[2])
            block = blocks.blkIndex[blk]
            k = block[1]
            t = block[2]
            for i=1:ngen
                Pg[i,k,t] = 0.5*(gens[i].Pmax + gens[i].Pmin)
            end
            for i=1:ngen
                Qg[i,k,t] = 0.5*(gens[i].Qmax + gens[i].Qmin)
            end
            for i=1:nbus
                Vm[i,k,t] = 0.5*(buses[i].Vmax + buses[i].Vmin)
                Va[i,k,t] = opfdata.buses[opfdata.bus_ref].Va
            end
            if T > 1
                for i=1:ngen
                    St[i,t] = gens[i].ramp_agc
                end
            end
            if K > 1
                for i=1:ngen
                    Sk[i,k,t] = gens[i].scen_agc
                end
            end
        end

        new(Pg,Qg,Vm,Va,ωt,St,Zt,Sk,Zk,sigma_real,sigma_imag,sigma_lineFr,sigma_lineTo)
    end
end

function OPFPrimalSolution(nlp::AbstractNLPEvaluator)
    return OPFPrimalSolution(
        nlp.opfdata,
        nlp.modelinfo,
        nlp.problem.opfBlockData,
        nlp.problem.blkLocalIndices,
        nlp.problem.blkLinkedIndices
    )
end

mutable struct OPFDualSolution <: AbstractDualSolution
    ramping
    ctgs

    function OPFDualSolution(
        opfdata::OPFData,
        modelinfo::ModelInfo,
        blocks::Union{AbstractBlocks,Nothing} = nothing,
        blkLocalIndices::Nothing = nothing,
        blkLinkedIndices::Nothing = nothing,
    )
        ngen = length(opfdata.generators)
        T = modelinfo.num_time_periods
        K = modelinfo.num_ctgs + 1 # base case counted separately
        @assert T >= 1
        @assert K >= 1

        ramping = zeros(ngen,T)
        ctgs = zeros(ngen,K,T)

        new(ramping,ctgs)
    end

    function OPFDualSolution(
        opfdata::OPFData,
        modelinfo::ModelInfo,
        blocks::AbstractBlocks,
        blkLocalIndices::Array{Int64},
        blkLinkedIndices::Array{Int64},
    )
        ngen = length(opfdata.generators)
        T = modelinfo.num_time_periods
        K = modelinfo.num_ctgs + 1 # base case counted separately
        @assert T >= 1
        @assert K >= 1

        blkindices = (blkLocalIndices, blkLinkedIndices)
        ramping = LocalStorage{1,ngen}(T, blocks, blkindices)
        ctgs = LocalStorage{2,ngen}(K, T, blocks, blkindices)

        new(ramping,ctgs)
    end
end

function OPFDualSolution(nlp::AbstractNLPEvaluator)
    return OPFDualSolution(
        nlp.opfdata,
        nlp.modelinfo,
        nlp.problem.opfBlockData,
        nlp.problem.blkLocalIndices,
        nlp.problem.blkLinkedIndices,
        )
end

function get_block_view(x::OPFPrimalSolution,
                        block::CartesianIndex,
                        modelinfo::ModelInfo,
                        algparams::AlgParams)
    k = block[1]
    t = block[2]
    range_k = algparams.decompCtgs ? (k:k) : (1:(modelinfo.num_ctgs + 1))

    Pg = x.Pg[:, range_k, t]
    Qg = x.Qg[:, range_k, t]
    Vm = x.Vm[:, range_k, t]
    Va = x.Va[:, range_k, t]
    ωt = x.ωt[range_k, t]
    St = x.St[:, t]
    Zt = x.Zt[:, t]
    Sk = x.Sk[:, range_k, t]
    Zk = x.Zk[:, range_k, t]

    return CatView(Pg, Qg, Vm, Va, ωt, St, Zt, Sk, Zk)
end

function get_coupling_view(x::OPFPrimalSolution,
                           modelinfo::ModelInfo,
                           algparams::AlgParams)
    Pg = @view x.Pg[:]
    return Pg
end

function get_coupling_view(λ::OPFDualSolution,
                           modelinfo::ModelInfo,
                           algparams::AlgParams)
    ramping = @view λ.ramping[:]
    ctgs = @view λ.ctgs[:]

    if algparams.decompCtgs
        return CatView(ramping, ctgs)
    else
        return CatView(ramping)
    end
end

function dist(x1::OPFPrimalSolution, x2::OPFPrimalSolution,
              modelinfo::ModelInfo,
              algparams::AlgParams,
              lnorm = Inf)
    x1vec = get_coupling_view(x1, modelinfo, algparams)
    x2vec = get_coupling_view(x2, modelinfo, algparams)
    return norm(x1vec .- x2vec, lnorm) #/(1e-16 + norm(x2vec, lnorm))
end

function dist(λ1::OPFDualSolution, λ2::OPFDualSolution,
              modelinfo::ModelInfo,
              algparams::AlgParams,
              lnorm = Inf)
    λ1vec = get_coupling_view(λ1, modelinfo, algparams)
    λ2vec = get_coupling_view(λ2, modelinfo, algparams)
    return norm(λ1vec .- λ2vec, lnorm) #/(1e-16 + norm(λ2vec, lnorm))
end
