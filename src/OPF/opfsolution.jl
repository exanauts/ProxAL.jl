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
    sigma

    function OPFPrimalSolution(opfdata::OPFData, modelinfo::ModelInfo) 
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
        sigma = zeros(K,T)

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

        new(Pg,Qg,Vm,Va,ωt,St,Zt,Sk,Zk,sigma)
    end
end

function OPFPrimalSolution(nlp::AbstractNLPEvaluator) 
    return OPFPrimalSolution(nlp.opfdata, nlp.modelinfo)
end

mutable struct OPFDualSolution <: AbstractDualSolution
    ramping
    ctgs

    function OPFDualSolution(opfdata::OPFData, modelinfo::ModelInfo) 
        ngen = length(opfdata.generators)
        T = modelinfo.num_time_periods
        K = modelinfo.num_ctgs + 1 # base case counted separately
        @assert T >= 1
        @assert K >= 1

        ramping = zeros(ngen,T)
        ctgs = zeros(ngen,K,T)

        new(ramping,ctgs)
    end
end

function OPFDualSolution(nlp::AbstractNLPEvaluator) 
    return OPFDualSolution(nlp.opfdata, nlp.modelinfo)
end

function get_block_view(x::OPFPrimalSolution,
                        block::CartesianIndex,
                        modelinfo::ModelInfo,
                        algparams::AlgParams)
    k = block[1]
    t = block[2]
    range_k = algparams.decompCtgs ? (k:k) : (1:(modelinfo.num_ctgs + 1))
    Pg = view(x.Pg, :, range_k, t)
    Qg = view(x.Qg, :, range_k, t)
    Vm = view(x.Vm, :, range_k, t)
    Va = view(x.Va, :, range_k, t)
    ωt = view(x.ωt, range_k, t)
    St = view(x.St, :, t)
    Zt = view(x.Zt, :, t)
    Sk = view(x.Sk, :, range_k, t)
    Zk = view(x.Zk, :, range_k, t)

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
