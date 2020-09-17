#
# Data structures to represent primal and dual solutions
#
mutable struct PrimalSolution
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

    function PrimalSolution(opfdata::OPFData; modelinfo::ModelParams = ModelParams())
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
end

mutable struct DualSolution
    ramping
    ramping_p
    ramping_n
    ctgs
    ctgs_p
    ctgs_n

    function DualSolution(opfdata::OPFData; modelinfo::ModelParams = ModelParams())
        ngen = length(opfdata.generators)
        T = modelinfo.num_time_periods
        K = modelinfo.num_ctgs + 1 # base case counted separately
        @assert T >= 1
        @assert K >= 1

        ramping = zeros(ngen,T)
        ramping_p = zeros(ngen,T)
        ramping_n = zeros(ngen,T)
        ctgs = zeros(ngen,K,T)
        ctgs_p = zeros(ngen,K,T)
        ctgs_n = zeros(ngen,K,T)

        new(ramping,ramping_p,ramping_n,ctgs,ctgs_p,ctgs_n)
    end
end

function get_block_view(x::PrimalSolution, block::CartesianIndex;
                        modelinfo::ModelParams,
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
    Zk = view(x.Sk, :, range_k, t)
    if modelinfo.allow_constr_infeas
        sigma_real = view(x.sigma_real, :, range_k, t)
        sigma_imag = view(x.sigma_imag, :, range_k, t)
        sigma_lineFr = view(x.sigma_lineFr, :, range_k, t)
        sigma_lineTo = view(x.sigma_lineTo, :, range_k, t)

        solution = CatView(Pg, Qg, Vm, Va, ωt, St, Zt, Sk, Zk, sigma_real, sigma_imag, sigma_lineFr, sigma_lineTo)
    else
        solution = CatView(Pg, Qg, Vm, Va, ωt, St, Zt, Sk, Zk)
    end

    return solution
end

function get_coupling_view(x::PrimalSolution;
                           modelinfo::ModelParams,
                           algparams::AlgParams)
    Pg = @view x.Pg[:]
    return Pg
    #=
    ωt = @view x.ωt[:]
    Zt = @view x.Zt[:]
    Zk = @view x.Sk[:]

    if algparams.decompCtgs
        if modelinfo.time_link_constr_type == :penalty
            if modelinfo.ctgs_link_constr_type == :frequency_ctrl
                return CatView(Pg, ωt, Zt)
            elseif modelinfo.ctgs_link_constr_type ∈ [:preventive_penalty, :corrective_penalty]
                return CatView(Pg, Zt, Zk)
            else
                return CatView(Pg, Zt)
            end
        else
            if modelinfo.ctgs_link_constr_type == :frequency_ctrl
                return CatView(Pg, ωt)
            elseif modelinfo.ctgs_link_constr_type ∈ [:preventive_penalty, :corrective_penalty]
                return CatView(Pg, Zk)
            else
                return CatView(Pg)
            end
        end
    else
        if modelinfo.time_link_constr_type == :penalty
            return CatView(Pg, Zt)
        else
            return CatView(Pg)
        end
    end
    =#
end

function get_coupling_view(λ::DualSolution;
                           modelinfo::ModelParams,
                           algparams::AlgParams)
    ramping = @view λ.ramping[:]
    ramping_p = @view λ.ramping_p[:]
    ramping_n = @view λ.ramping_n[:]
    ctgs = @view λ.ctgs[:]
    ctgs_p = @view λ.ctgs_p[:]
    ctgs_n = @view λ.ctgs_n[:]

    if algparams.decompCtgs
        if modelinfo.time_link_constr_type == :inequality
            if modelinfo.ctgs_link_constr_type == :corrective_inequality
                return CatView(ramping_p, ramping_n, ctgs_p, ctgs_n)
            else
                return CatView(ramping_p, ramping_n, ctgs)
            end
        else
            if modelinfo.ctgs_link_constr_type == :corrective_inequality
                return CatView(ramping, ctgs_p, ctgs_n)
            else
                return CatView(ramping, ctgs)
            end
        end
    else
        if modelinfo.time_link_constr_type == :inequality
            return CatView(ramping_p, ramping_n)
        else
            return CatView(ramping)
        end
    end
end

function dist(x1::PrimalSolution, x2::PrimalSolution;
              modelinfo::ModelParams,
              algparams::AlgParams,
              lnorm = Inf)
    x1vec = get_coupling_view(x1; modelinfo = modelinfo, algparams = algparams)
    x2vec = get_coupling_view(x2; modelinfo = modelinfo, algparams = algparams)
    return norm(x1vec .- x2vec, lnorm) #/(1e-16 + norm(x2vec, lnorm))
end

function dist(λ1::DualSolution, λ2::DualSolution;
              modelinfo::ModelParams,
              algparams::AlgParams,
              lnorm = Inf)
    λ1vec = get_coupling_view(λ1; modelinfo = modelinfo, algparams = algparams)
    λ2vec = get_coupling_view(λ2; modelinfo = modelinfo, algparams = algparams)
    return norm(λ1vec .- λ2vec, lnorm) #/(1e-16 + norm(λ2vec, lnorm))
end
