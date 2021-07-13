# FIX ME: A lot of the functions here are for the deprecated NonDecomposedModel
function opf_model_add_variables(opfmodel::JuMP.Model, opfdata::OPFData,
                                 modelinfo::ModelParams,
                                 algparams::AlgParams)
    # shortcuts for compactness
    buses = opfdata.buses
    gens  = opfdata.generators
    ngen  = length(gens)
    nbus  = length(buses)
    nline = length(opfdata.lines)
    T = modelinfo.num_time_periods
    K = modelinfo.num_ctgs + 1 # base case counted separately
    @assert T >= 1
    @assert K >= 1


    # Variables
    @variable(opfmodel, Pg[1:ngen,1:K,1:T])
    @variable(opfmodel, Qg[1:ngen,1:K,1:T])
    @variable(opfmodel, Vm[1:nbus,1:K,1:T])
    @variable(opfmodel, Va[1:nbus,1:K,1:T])
    @variable(opfmodel, ωt[1:K,1:T])        # frequency
    @variable(opfmodel, St[1:ngen,1:T])     # slack for ramp constraints
    @variable(opfmodel, Zt[1:ngen,1:T])     # slack for ramp constraints to guarantee ADMM converges
    @variable(opfmodel, Sk[1:ngen,1:K,1:T]) # slack for ctgs_link_constr
    @variable(opfmodel, Zk[1:ngen,1:K,1:T]) # slack for ctgs_link_constr to guarantee ADMM converges
    if modelinfo.allow_constr_infeas
        @variable(opfmodel, sigma_real[1:nbus,1:K,1:T], start = 0)
        @variable(opfmodel, sigma_imag[1:nbus,1:K,1:T], start = 0)
        @variable(opfmodel, sigma_lineFr[1:nline,1:K,1:T] >= 0, start = 0)
        @variable(opfmodel, sigma_lineTo[1:nline,1:K,1:T] >= 0, start = 0)
    end

    # Variable bounds and start values
    for i=1:ngen
        set_lower_bound.(Pg[i,:,:], gens[i].Pmin)
        set_upper_bound.(Pg[i,:,:], gens[i].Pmax)
        set_start_value.(Pg[i,:,:], 0.5*(gens[i].Pmax + gens[i].Pmin))
        set_lower_bound.(Qg[i,:,:], gens[i].Qmin)
        set_upper_bound.(Qg[i,:,:], gens[i].Qmax)
        set_start_value.(Qg[i,:,:], 0.5*(gens[i].Qmax + gens[i].Qmin))
    end

    fix.(Va[opfdata.bus_ref,:,:], buses[opfdata.bus_ref].Va)
    for i=1:nbus
        set_lower_bound.(Vm[i,:,:], buses[i].Vmin)
        set_upper_bound.(Vm[i,:,:], buses[i].Vmax)
        set_start_value.(Vm[i,:,:], 0.5*(buses[i].Vmax + buses[i].Vmin))
        set_start_value.(Va[i,:,:], buses[opfdata.bus_ref].Va)
        if modelinfo.allow_constr_infeas
            for k=1:K
                set_lower_bound.(sigma_real[i,k,:],-opfdata.Pd[i,:])
                set_upper_bound.(sigma_real[i,k,:], opfdata.Pd[i,:])
                set_lower_bound.(sigma_imag[i,k,:],-opfdata.Qd[i,:])
                set_upper_bound.(sigma_imag[i,k,:], opfdata.Qd[i,:])
            end
        end
    end
    fix.(ωt[1,:], 0)
    fix.(Zt[:,1], 0)
    fix.(Zk[:,1,:], 0)
    set_start_value.(Zt, 0)
    set_start_value.(Zk, 0)
    for k=2:K
        set_lower_bound.(ωt[k,:], -1)
        set_upper_bound.(ωt[k,:], +1)
        set_start_value.(ωt[k,:], 0)
    end
    for i=1:ngen
        set_lower_bound.(St[i,:], 0)
        set_upper_bound.(St[i,:], 2gens[i].ramp_agc)
        set_start_value.(St[i,:],  gens[i].ramp_agc)
        set_lower_bound.(Sk[i,:,:], 0)
        set_upper_bound.(Sk[i,:,:], 2gens[i].scen_agc)
        set_start_value.(Sk[i,:,:],  gens[i].scen_agc)
    end
    if algparams.mode ∈ [:nondecomposed, :lyapunov_bound]
        fix.(St[:,1], 0; force = true)
        fix.(Sk[:,1,:], 0; force = true)
    end
    if !algparams.decompCtgs
        fix.(Sk[:,1,:], 0; force = true)
    end
    if T > 1
        if modelinfo.time_link_constr_type == :inequality && algparams.mode == :nondecomposed
            fix.(St, 0; force = true)
        end
        if modelinfo.time_link_constr_type != :penalty
            fix.(Zt, 0; force = true)
        end
    end
    if K > 1
        if modelinfo.ctgs_link_constr_type == :frequency_ctrl
            fix.(Sk, 0; force = true)
            fix.(Zk, 0; force = true)
        else
            fix.(ωt, 0; force = true)
            if modelinfo.ctgs_link_constr_type ∈ [:preventive_equality, :preventive_penalty] ||
                (modelinfo.ctgs_link_constr_type == :corrective_inequality && (algparams.mode == :nondecomposed || !algparams.decompCtgs))
                fix.(Sk, 0; force = true)
            end
            if modelinfo.ctgs_link_constr_type ∉ [:preventive_penalty, :corrective_penalty]
                fix.(Zk, 0; force = true)
            end
        end
    end
end

function opf_model_add_block_constraints(opfmodel::JuMP.Model, opfdata::OPFData, rawdata::RawData, modelinfo::ModelParams)
    T = modelinfo.num_time_periods
    K = (modelinfo.num_ctgs + 1)
    if modelinfo.allow_constr_infeas
        @views for t=1:T, k=1:K
            opfdata_c = (k == 1) ? opfdata :
                opf_loaddata(rawdata; lineOff = opfdata.lines[rawdata.ctgs_arr[k - 1]], time_horizon_start = t, time_horizon_end = t, load_scale = modelinfo.load_scale, ramp_scale = modelinfo.ramp_scale)
            opf_model_add_real_power_balance_constraints(opfmodel, opfdata_c, opfmodel[:Pg][:,k,t], opfdata.Pd[:,t], opfmodel[:Vm][:,k,t], opfmodel[:Va][:,k,t], opfmodel[:sigma_real][:,k,t])
            opf_model_add_imag_power_balance_constraints(opfmodel, opfdata_c, opfmodel[:Qg][:,k,t], opfdata.Qd[:,t], opfmodel[:Vm][:,k,t], opfmodel[:Va][:,k,t], opfmodel[:sigma_imag][:,k,t])
            opf_model_add_line_power_constraints(opfmodel, opfdata_c, opfmodel[:Vm][:,k,t], opfmodel[:Va][:,k,t], opfmodel[:sigma_lineFr][:,k,t], opfmodel[:sigma_lineTo][:,k,t])
        end
    else
        zb = zeros(length(opfdata.buses))
        zl = zeros(length(opfdata.lines))
        @views for t=1:T, k=1:K
            opfdata_c = (k == 1) ? opfdata :
                opf_loaddata(rawdata; lineOff = opfdata.lines[rawdata.ctgs_arr[k - 1]], time_horizon_start = t, time_horizon_end = t, load_scale = modelinfo.load_scale, ramp_scale = modelinfo.ramp_scale)
            opf_model_add_real_power_balance_constraints(opfmodel, opfdata_c, opfmodel[:Pg][:,k,t], opfdata.Pd[:,t], opfmodel[:Vm][:,k,t], opfmodel[:Va][:,k,t], zb)
            opf_model_add_imag_power_balance_constraints(opfmodel, opfdata_c, opfmodel[:Qg][:,k,t], opfdata.Qd[:,t], opfmodel[:Vm][:,k,t], opfmodel[:Va][:,k,t], zb)
            opf_model_add_line_power_constraints(opfmodel, opfdata_c, opfmodel[:Vm][:,k,t], opfmodel[:Va][:,k,t], zl, zl)
        end
    end
    return nothing
end

function opf_model_add_line_power_constraints(opfmodel::JuMP.Model, opfdata::OPFData, Vm, Va, sigma_lineFr, sigma_lineTo)
    # Network data short-hands
    baseMVA = opfdata.baseMVA
    buses = opfdata.buses
    lines = opfdata.lines
    busIdx = opfdata.BusIdx
    YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI = computeAdmitances(lines, buses, baseMVA)

    for l in 1:length(lines)
        line = lines[l]
        from = busIdx[line.from]
        to = busIdx[line.to]
        if line.rateA ≈ 0 || line.rateA >= 1.0e10
            continue
        end
        flowmax=(line.rateA/baseMVA)^2

        #branch apparent power limits (from bus)
        Yff_abs2=YffR[l]^2+YffI[l]^2; Yft_abs2=YftR[l]^2+YftI[l]^2
        Yre=YffR[l]*YftR[l]+YffI[l]*YftI[l]; Yim=-YffR[l]*YftI[l]+YffI[l]*YftR[l]
        @NLconstraint(opfmodel,
            Vm[from]^2 *
            ( Yff_abs2*Vm[from]^2 + Yft_abs2*Vm[to]^2
            + 2*Vm[from]*Vm[to]*(Yre*cos(Va[from]-Va[to])-Yim*sin(Va[from]-Va[to]))
            )
            - flowmax
            - (sigma_lineFr[l]/baseMVA)
            <=0
        )

        #branch apparent power limits (to bus)
        Ytf_abs2=YtfR[l]^2+YtfI[l]^2; Ytt_abs2=YttR[l]^2+YttI[l]^2
        Yre=YtfR[l]*YttR[l]+YtfI[l]*YttI[l]; Yim=-YtfR[l]*YttI[l]+YtfI[l]*YttR[l]
        @NLconstraint(opfmodel,
            Vm[to]^2 *
            ( Ytf_abs2*Vm[from]^2 + Ytt_abs2*Vm[to]^2
            + 2*Vm[from]*Vm[to]*(Yre*cos(Va[from]-Va[to])-Yim*sin(Va[from]-Va[to]))
            )
            - flowmax
            - (sigma_lineTo[l]/baseMVA)
            <=0
        )
    end
end

function opf_model_add_real_power_balance_constraints(opfmodel::JuMP.Model, opfdata::OPFData, Pg, Pd, Vm, Va, sigma_real)
    # Network data short-hands
    baseMVA = opfdata.baseMVA
    buses = opfdata.buses
    busIdx = opfdata.BusIdx
    BusGeners = opfdata.BusGenerators
    Ybus = opfdata.Ybus

    rows = Ybus.rowval
    yvals = Ybus.nzval
    g_ij = real.(yvals)
    b_ij = imag.(yvals)

    # Power Balance Equations
    for b in 1:length(buses)
        #real part
        b_start, b_end = Ybus.colptr[b], Ybus.colptr[b+1]-1
        @NLconstraint(opfmodel,
            Vm[b] * sum(Vm[rows[c]] * (g_ij[c] * cos(Va[b] - Va[rows[c]]) + b_ij[c] * sin(Va[b] - Va[rows[c]])) for c in b_start:b_end)
            - ( sum(baseMVA*Pg[g] for g in get(BusGeners, b, Int[])) - Pd[b] + sigma_real[b]) / baseMVA      # Sbus part
            ==0
        )
    end
end

function opf_model_add_imag_power_balance_constraints(opfmodel::JuMP.Model, opfdata::OPFData, Qg, Qd, Vm, Va, sigma_imag)
    # Network data short-hands
    baseMVA = opfdata.baseMVA
    buses = opfdata.buses
    busIdx = opfdata.BusIdx
    BusGeners = opfdata.BusGenerators
    Ybus = opfdata.Ybus

    rows = Ybus.rowval
    yvals = Ybus.nzval
    g_ij = real.(yvals)
    b_ij = imag.(yvals)

    # Power Balance Equations
    for b in 1:length(buses)
        #imaginary part
        b_start, b_end = Ybus.colptr[b], Ybus.colptr[b+1]-1
        @NLconstraint(opfmodel,
            Vm[b] * sum(Vm[rows[c]] * (g_ij[c] * sin(Va[b] - Va[rows[c]]) - b_ij[c] * cos(Va[b] - Va[rows[c]])) for c in b_start:b_end)
            - ( sum(baseMVA*Qg[g] for g in get(BusGeners, b, Int[])) - Qd[b] + sigma_imag[b]) / baseMVA      #Sbus part
            ==0
        )
    end
end

function opf_model_add_time_linking_constraints(opfmodel::JuMP.Model, opfdata::OPFData, modelinfo::ModelParams)
    (ngen, K, T) = size(opfmodel[:Pg])

    if T > 1
        link = compute_time_linking_constraints(opfmodel, opfdata, modelinfo)

        if modelinfo.time_link_constr_type == :inequality
            @constraint(opfmodel, ramping_p[g=1:ngen,t=2:T], link[:ramping_p][g,t] <= 0)
            @constraint(opfmodel, ramping_n[g=1:ngen,t=2:T], link[:ramping_n][g,t] <= 0)
        else
            @constraint(opfmodel, ramping[g=1:ngen,t=2:T], link[:ramping][g,t] == 0)
        end
    end

    return nothing
end

function opf_model_add_ctgs_linking_constraints(opfmodel::JuMP.Model, opfdata::OPFData, modelinfo::ModelParams)
    (ngen, K, T) = size(opfmodel[:Pg])

    if K > 1
        link = compute_ctgs_linking_constraints(opfmodel, opfdata, modelinfo)

        if modelinfo.ctgs_link_constr_type == :corrective_inequality
            @constraint(opfmodel, ctgs_p[g=1:ngen,k=2:K,t=1:T], link[:ctgs_p][g,k,t] <= 0)
            @constraint(opfmodel, ctgs_n[g=1:ngen,k=2:K,t=1:T], link[:ctgs_p][g,k,t] <= 0)
        else
            @constraint(opfmodel, ctgs[g=1:ngen,k=2:K,t=1:T], link[:ctgs][g,k,t] == 0)
        end
    end

    return nothing
end

function compute_objective_function(opfdict, opfdata::OPFData, modelinfo::ModelParams, algparams::AlgParams, k::Int, t::Int)
    Pg = opfdict[:Pg]
    ωt = opfdict[:ωt]
    Zt = opfdict[:Zt]
    Zk = opfdict[:Zk]

    (ngen, K, T) = size(Pg)
    @assert t >= 1 && t <= T
    @assert k >= 1 && k <= K

    if k == 1 && modelinfo.allow_obj_gencost
        baseMVA = opfdata.baseMVA
        gens = opfdata.generators
        obj_gencost = sum(gens[g].coeff[gens[g].n-2]*(baseMVA*Pg[g,k,t])^2
                        + gens[g].coeff[gens[g].n-1]*(baseMVA*Pg[g,k,t])
                        + gens[g].coeff[gens[g].n  ]
                        for g=1:ngen)
    else
        obj_gencost = 0
    end
    if modelinfo.allow_constr_infeas
        sigma_real = opfdict[:sigma_real]
        sigma_imag = opfdict[:sigma_imag]
        sigma_lineFr = opfdict[:sigma_lineFr]
        sigma_lineTo = opfdict[:sigma_lineTo]
        nbus = size(sigma_real, 1)
        nline = size(sigma_lineFr, 1)
        obj_constr_infeas = 0.5sum(sigma_real[b,k,t]^2 + sigma_imag[b,k,t]^2 for b=1:nbus) +
                            0.5sum(sigma_lineFrom[l,k,t]^2 + sigma_lineTo[l,k,t]^2 for l=1:nline)
        obj_constr_infeas = (k > 1 ? modelinfo.weight_ctgs : 1.0)*obj_constr_infeas
    else
        obj_constr_infeas = 0
    end
    if modelinfo.ctgs_link_constr_type == :frequency_ctrl
        obj_freq_ctrl = 0.5*ωt[k,t]^2
    else
        obj_freq_ctrl = 0
    end
    if modelinfo.time_link_constr_type == :penalty
        obj_bigM_penalty_time = 0.5sum(Zt[:,t].^2)
    else
        obj_bigM_penalty_time = 0
    end
    if modelinfo.ctgs_link_constr_type ∈ [:preventive_penalty, :corrective_penalty]
        obj_bigM_penalty_ctgs = 0.5sum(Zk[:,k,t].^2)
    else
        obj_bigM_penalty_ctgs = 0
    end

    return modelinfo.obj_scale*(
            obj_gencost +
            (modelinfo.weight_constr_infeas*obj_constr_infeas) +
            (modelinfo.weight_freq_ctrl*obj_freq_ctrl) +
            (algparams.θ_t*obj_bigM_penalty_time) +
            (algparams.θ_c*obj_bigM_penalty_ctgs)
        )
end

function compute_time_linking_constraints(opfdict, opfdata::OPFData, modelinfo::ModelParams, tIdx::Int = 0)
    Pg = opfdict[:Pg]
    St = opfdict[:St]
    Zt = opfdict[:Zt]

    gens = opfdata.generators
    (ngen, K, T) = size(Pg)
    Trange = (tIdx == 0) ? (1:T) : (tIdx:tIdx)

    link = Dict{Symbol, Array}()
    singletons = []
    if modelinfo.time_link_constr_type == :inequality
        link[:ramping_p] = [(t > 1) ? (+Pg[g,1,t-1] - Pg[g,1,t] - gens[g].ramp_agc) : 0.0 for g=1:ngen,t=Trange]
        link[:ramping_n] = [(t > 1) ? (-Pg[g,1,t-1] + Pg[g,1,t] - gens[g].ramp_agc) : 0.0 for g=1:ngen,t=Trange]
    else
        link[:ramping] = [(t > 1) ? (Pg[g,1,t-1] - Pg[g,1,t] + St[g,t] + Zt[g,t] - gens[g].ramp_agc) : 0.0 for g=1:ngen,t=Trange]
    end

    return link
end

function compute_ctgs_linking_constraints(opfdict, opfdata::OPFData, modelinfo::ModelParams, kIdx::Int = 0, tIdx::Int = 0)
    Pg = opfdict[:Pg]
    ωt = opfdict[:ωt]
    Sk = opfdict[:Sk]
    Zk = opfdict[:Zk]

    gens = opfdata.generators
    (ngen, K, T) = size(Pg)
    Trange = (tIdx == 0) ? (1:T) : (tIdx:tIdx)
    Krange = (kIdx == 0) ? (1:K) : (kIdx:kIdx)


    link = Dict{Symbol, Array}()
    if modelinfo.ctgs_link_constr_type == :corrective_inequality
        link[:ctgs_p] = [(k > 1) ? (+Pg[g,1,t] - Pg[g,k,t] - gens[g].scen_agc) : 0.0 for g=1:ngen,k=Krange,t=Trange]
        link[:ctgs_n] = [(k > 1) ? (-Pg[g,1,t] + Pg[g,k,t] - gens[g].scen_agc) : 0.0 for g=1:ngen,k=Krange,t=Trange]
    else
        β = [gens[g].scen_agc for g=1:ngen]
        (modelinfo.ctgs_link_constr_type ∉ [:corrective_inequality, :corrective_equality, :corrective_penalty]) && (β .= 0)
        link[:ctgs] = [(k > 1) ? (Pg[g,1,t] - Pg[g,k,t] + (gens[g].alpha*ωt[k,t]) + Sk[g,k,t] + Zk[g,k,t] - β[g]) : 0.0 for g=1:ngen,k=Krange,t=Trange]
    end

    return link
end

function compute_quadratic_penalty(opfdict, opfdata::OPFData,
                                   opfBlockData::OPFBlocks, blk::Int,
                                   modelinfo::ModelParams, algparams::AlgParams)
    (ngen, K, T) = size(opfdict[:Pg])
    block = opfBlockData.blkIndex[blk]
    k = block[1]
    t = block[2]

    if t > 1
        inequality = (modelinfo.time_link_constr_type == :inequality)
        inequality && (modelinfo.time_link_constr_type = :equality)
        link = compute_time_linking_constraints(opfdict, opfdata, modelinfo, t)
        inequality && (modelinfo.time_link_constr_type = :inequality)
        lyapunov_quadratic_penalty_time = sum(link[:ramping][:].^2)
    else
        lyapunov_quadratic_penalty_time = 0
    end


    if k > 1 && algparams.decompCtgs
        inequality = (modelinfo.ctgs_link_constr_type == :corrective_inequality)
        inequality && (modelinfo.ctgs_link_constr_type = :corrective_equality)
        link = compute_ctgs_linking_constraints(opfdict, opfdata, modelinfo, k, t)
        inequality && (modelinfo.ctgs_link_constr_type = :corrective_inequality)

        lyapunov_quadratic_penalty_ctgs = sum(link[:ctgs][:].^2)
    else
        lyapunov_quadratic_penalty_ctgs = 0
    end


    return ((0.5algparams.ρ_t*lyapunov_quadratic_penalty_time) +
            (0.5algparams.ρ_c*lyapunov_quadratic_penalty_ctgs))
end

function compute_quadratic_penalty(opfdict, opfdata::OPFData,
                                   modelinfo::ModelParams, algparams::AlgParams)
    (ngen, K, T) = size(opfdict[:Pg])

    if T > 1
        inequality = (modelinfo.time_link_constr_type == :inequality)
        inequality && (modelinfo.time_link_constr_type = :equality)
        link = compute_time_linking_constraints(opfdict, opfdata, modelinfo)
        inequality && (modelinfo.time_link_constr_type = :inequality)

        lyapunov_quadratic_penalty_time = sum(link[:ramping][:,2:T].^2)
    else
        lyapunov_quadratic_penalty_time = 0
    end


    if K > 1 && algparams.decompCtgs
        inequality = (modelinfo.ctgs_link_constr_type == :corrective_inequality)
        inequality && (modelinfo.ctgs_link_constr_type = :corrective_equality)
        link = compute_ctgs_linking_constraints(opfdict, opfdata, modelinfo)
        inequality && (modelinfo.ctgs_link_constr_type = :corrective_inequality)

        lyapunov_quadratic_penalty_ctgs = sum(link[:ctgs][:,2:K,:].^2)
    else
        lyapunov_quadratic_penalty_ctgs = 0
    end


    return ((0.5algparams.ρ_t*lyapunov_quadratic_penalty_time) +
            (0.5algparams.ρ_c*lyapunov_quadratic_penalty_ctgs))
end

function compute_lagrangian_function(opfdict, λ::DualSolution, opfdata::OPFData,
                                     opfBlockData::OPFBlocks, blk::Int,
                                     modelinfo::ModelParams, algparams::AlgParams)

    (ngen, K, T) = size(opfdict[:Pg])
    block = opfBlockData.blkIndex[blk]
    k = block[1]
    t = block[2]

    obj = compute_objective_function(opfdict, opfdata, opfBlockData, blk, modelinfo, algparams)

    if T > 1
        @assert modelinfo.time_link_constr_type == :penalty
        link = compute_time_linking_constraints(opfdict, opfdata, modelinfo, t)
        lagrangian_t = sum(λ.ramping[:,t].*link[:ramping][:])
    else
        lagrangian_t = 0
    end


    if K > 1 && algparams.decompCtgs
        @assert modelinfo.ctgs_link_constr_type ∈ [:frequency_ctrl, :preventive_penalty, :corrective_penalty]
        link = compute_ctgs_linking_constraints(opfdict, opfdata, modelinfo, k, t)
        lagrangian_c = sum(λ.ctgs[:,k,t].*link[:ctgs][:])
    else
        lagrangian_c = 0
    end

    return obj + lagrangian_t + lagrangian_c
end

function compute_proximal_function(x1::PrimalSolution, x2::PrimalSolution,
                                   opfBlockData::OPFBlocks, blk::Int,
                                   modelinfo::ModelParams, algparams::AlgParams)
    block = opfBlockData.blkIndex[blk]
    k = block[1]
    t = block[2]

    prox_pg = sum((x1.Pg[:,1,t] .- x2.Pg[:,1,t]).^2)
    prox_penalty = 0

    if modelinfo.time_link_constr_type == :penalty
        if t > 1
            prox_penalty += sum((x1.Zt[:,t] .- x2.Zt[:,t]).^2)
        end
    end

    if k > 1 && algparams.decompCtgs
        prox_pg += sum((x1.Pg[:,k,t] .- x2.Pg[:,k,t]).^2)

        if modelinfo.ctgs_link_constr_type == :frequency_ctrl
            prox_penalty += (x1.ωt[k,t] - x2.ωt[k,t])^2
        end
        if modelinfo.ctgs_link_constr_type ∈ [:preventive_penalty, :corrective_penalty]
            prox_penalty += sum((x1.Zk[:,k,t] .- x2.Zk[:,k,t]).^2)
        end
    end

    return 0.5algparams.τ*(prox_pg + prox_penalty)
end

function compute_objective_function(opfdict, opfdata::OPFData, opfBlockData::OPFBlocks, blk::Int, modelinfo::ModelParams, algparams::AlgParams)
    K = size(opfdict[:Pg],2)
    block = opfBlockData.blkIndex[blk]
    t = block[2]
    if K > 1 && algparams.decompCtgs
        k = block[1]
        return compute_objective_function(opfdict, opfdata, modelinfo, algparams, k, t)
    else
        return sum(compute_objective_function(opfdict, opfdata, modelinfo, algparams, k, t) for k=1:K)
    end
end

function compute_objective_function(opfdict, opfdata::OPFData, modelinfo::ModelParams, algparams::AlgParams)
    (_, K, T) = size(opfdict[:Pg])
    return sum(compute_objective_function(opfdict, opfdata, modelinfo, algparams, k, t) for k=1:K, t=1:T)
end

function compute_objective_function(x::PrimalSolution, opfdata::OPFData,
                                    opfBlockData::OPFBlocks, blk::Int,
                                    modelinfo::ModelParams, algparams::AlgParams)
    d = Dict(:Pg => x.Pg,
             :ωt => x.ωt,
             :Zt => x.Zt,
             :Zk => x.Zk,
             :sigma_real => x.sigma_real,
             :sigma_imag => x.sigma_imag,
             :sigma_lineFr => x.sigma_lineFr,
             :sigma_lineTo => x.sigma_lineTo)
    return compute_objective_function(d, opfdata, opfBlockData, blk, modelinfo, algparams)
end

function compute_lyapunov_function(x::PrimalSolution, λ::DualSolution, opfdata::OPFData,
                                   opfBlockData::OPFBlocks, blk::Int,
                                   xref::PrimalSolution,
                                   modelinfo::ModelParams,
                                   algparams::AlgParams)
    d = Dict(:Pg => x.Pg,
             :ωt => x.ωt,
             :St => x.St,
             :Zt => x.Zt,
             :Sk => x.Sk,
             :Zk => x.Zk,
             :sigma_real => x.sigma_real,
             :sigma_imag => x.sigma_imag,
             :sigma_lineFr => x.sigma_lineFr,
             :sigma_lineTo => x.sigma_lineTo)
    lagrangian = compute_lagrangian_function(d, λ, opfdata, opfBlockData, blk, modelinfo, algparams)
    quadratic_penalty = compute_quadratic_penalty(d, opfdata, opfBlockData, blk, modelinfo, algparams)
    # proximal = 0.5algparams.τ*dist(x, xref; modelinfo = modelinfo, algparams = algparams, lnorm = 2)^2
    proximal = compute_proximal_function(x, xref, opfBlockData, blk, modelinfo, algparams)

    return lagrangian + quadratic_penalty + 0.5proximal
end

function compute_dual_error(x::PrimalSolution, xprev::PrimalSolution, λ::DualSolution, λprev::DualSolution, opfdata::OPFData,
                            opfBlockData::OPFBlocks, blk::Int,
                            modelinfo::ModelParams,
                            algparams::AlgParams;
                            lnorm = Inf)
    (ngen, K, T) = size(x.Pg)
    block = opfBlockData.blkIndex[blk]
    k = block[1]
    t = block[2]

    err_pg = zeros(ngen, K)
    err_ωt = zeros(K)
    err_st = zeros(ngen)
    err_zt = zeros(ngen)
    err_sk = zeros(ngen, K)
    err_zk = zeros(ngen, K)

    if T > 1
        @assert modelinfo.time_link_constr_type == :penalty
        @views begin
            # for convenience
            β = zeros(ngen)
            for g=1:ngen
                β[g] = opfdata.generators[g].ramp_agc
            end

            #----------------------------------------------------------------------------
            true_pg_dual = zeros(ngen, K)
            lagrangian_pg_err = zeros(ngen, K)
            penalty_pg_forward_err = zeros(ngen, K)
            penalty_pg_reverse_err = zeros(ngen, K)
            if t > 1
                true_pg_dual = -λ.ramping[:,t]
                lagrangian_pg_err = -λprev.ramping[:,t]
                penalty_pg_forward_err = -algparams.ρ_t[:,t].*(
                                            (algparams.jacobi ? xprev.Pg[:,1,t-1] : x.Pg[:,1,t-1]) .-
                                            x.Pg[:,1,t] .+ x.St[:,t] .+ xprev.Zt[:,t] .- β[:]
                                        )
                penalty_pg_reverse_err = +algparams.ρ_t[:,t].*(
                                            x.Pg[:,1,t-1] .- xprev.Pg[:,1,t] .+ xprev.St[:,t] .+ xprev.Zt[:,t] .- β[:]
                                        )
            end
            prox_pg_err = algparams.τ*(x.Pg[:,1,t] .- xprev.Pg[:,1,t])
            #----------------------------------------------------------------------------

            if t > 1
                err_pg[:,1] += -true_pg_dual .+ lagrangian_pg_err .- penalty_pg_reverse_err
                err_pg[:,1] += true_pg_dual .- lagrangian_pg_err .- penalty_pg_forward_err
            end
            err_pg[:,1] -= prox_pg_err[:]
            if t > 1
                err_st[:] += penalty_pg_forward_err
                err_st[:] += -true_pg_dual .+ lagrangian_pg_err
                err_zt[:] += -true_pg_dual .+ lagrangian_pg_err - (algparams.ρ_t[:,t].*(
                                            x.Pg[:,1,t-1] .- x.Pg[:,1,t] .+ x.St[:,t] .+ x.Zt[:,t] .- β
                                        ))
                err_zt[:] -= (algparams.τ*(x.Zt[:,t] - xprev.Zt[:,t]))
            end
        end
    end

    if K > 1 && algparams.decompCtgs
        @assert modelinfo.ctgs_link_constr_type ∈ [:frequency_ctrl, :preventive_penalty, :corrective_penalty]
        @views begin
            # for convenience
            β = zeros(ngen, K, T)
            pg_base = zeros(ngen, K, T)
            pg_base_prev = zeros(ngen, K, T)
            if modelinfo.ctgs_link_constr_type ∈ [:corrective_inequality, :corrective_equality, :corrective_penalty]
                for g=1:ngen
                    β[g,:,:] .= opfdata.generators[g].scen_agc
                end
            end
            for k=2:K, g=1:ngen
                pg_base[g,k,:] .= x.Pg[g,1,:]
                pg_base_prev[g,k,:] .= xprev.Pg[g,1,:]
            end

            #----------------------------------------------------------------------------
            true_pg_ctgs_dual = -λ.ctgs[:,2:K,:]
            lagrangian_pg_ctgs_err = -λprev.ctgs[:,2:K,:]
            penalty_pg_base_err = pg_base[:,2:K,:] .-
                                        xprev.Pg[:,2:K,:] .+ xprev.Sk[:,2:K,:] .+ xprev.Zk[:,2:K,:] .- β[:,2:K,:]
            penalty_pg_ctgs_err = (algparams.jacobi ? pg_base_prev[:,2:K,:] : pg_base[:,2:K,:]) -
                                            x.Pg[:,2:K,:] .+     x.Sk[:,2:K,:] .+ xprev.Zk[:,2:K,:] .- β[:,2:K,:]
            prox_pg_err = algparams.τ*(x.Pg[:,2:K,:] .- xprev.Pg[:,2:K,:])
            for g=1:ngen
                penalty_pg_ctgs_err[g,:,:] += opfdata.generators[g].alpha*xprev.ωt[2:K,:]
                penalty_pg_base_err[g,:,:] += opfdata.generators[g].alpha*xprev.ωt[2:K,:]
            end
            penalty_pg_base_err .= algparams.ρ_c[:,2:K,:].*penalty_pg_base_err
            penalty_pg_ctgs_err .= -algparams.ρ_c[:,2:K,:].*penalty_pg_ctgs_err
            #----------------------------------------------------------------------------

            err_pg[:,1,:] += dropdims(sum(-true_pg_ctgs_dual .+ lagrangian_pg_ctgs_err .- penalty_pg_base_err; dims = 2); dims = 2)
            err_pg[:,2:K,:] += true_pg_ctgs_dual .- lagrangian_pg_ctgs_err .- penalty_pg_ctgs_err .- prox_pg_err
            if modelinfo.ctgs_link_constr_type == :frequency_ctrl
                for g=1:ngen
                    err_ωt[2:K,:] += opfdata.generators[g].alpha*(
                                        -true_pg_ctgs_dual[g,:,:].+lagrangian_pg_ctgs_err[g,:,:].-
                                        (algparams.ρ_c[g,2:K,:].*(
                                            pg_base[g,2:K,:] .- x.Pg[g,2:K,:] .+ (opfdata.generators[g].alpha*x.ωt[2:K,:])
                                        ))
                                    )
                    err_ωt[2:K,:] -= algparams.τ*(x.ωt[2:K,:] - xprev.ωt[2:K,:])
                end
            else
                err_sk[:,2:K,:] += penalty_pg_ctgs_err
            end
            if modelinfo.time_link_constr_type ∈ [:preventive_equality, :preventive_penalty, :corrective_equality, :corrective_penalty]
                err_sk[:,2:K,:] += -true_pg_ctgs_dual .+ lagrangian_pg_ctgs_err
            end
            if modelinfo.time_link_constr_type ∈ [:preventive_penalty, :corrective_penalty]
                err_zk[:,2:K,:] += -true_pg_ctgs_dual .+ lagrangian_pg_ctgs_err - (algparams.ρ_c[:,2:K,:].*(
                                            pg_base .- x.Pg[:,2:K,:] .+ x.Sk[:,2:K,:] .+ x.Zk[:,2:K,:] .- β[:,2:K,:]
                                        ))
                err_zk[:,2:K,:] -= (algparams.τ*(x.Zk[:,2:K,:] - xprev.Zk[:,2:K,:]))
            end
        end
    end

    err_pg_view = view(err_pg, :, :, :)
    err_ωt_view = view(err_ωt, :, :)
    err_st_view = view(err_st, :, :)
    err_zt_view = view(err_zt, :, :)
    err_sk_view = view(err_sk, :, :, :)
    err_zk_view = view(err_zk, :, :, :)

    dual_error = CatView(err_pg_view, err_ωt_view, err_st_view, err_zt_view, err_sk_view, err_zk_view)

    return dual_error
end

function compute_dual_error(x::PrimalSolution, xprev::PrimalSolution, λ::DualSolution, λprev::DualSolution, opfdata::OPFData,
                            modelinfo::ModelParams,
                            algparams::AlgParams;
                            lnorm = Inf)
    (ngen, K, T) = size(x.Pg)

    err_pg = zeros(ngen, K, T)
    err_ωt = zeros(K, T)
    err_st = zeros(ngen, T)
    err_zt = zeros(ngen, T)
    err_sk = zeros(ngen, K, T)
    err_zk = zeros(ngen, K, T)

    if T > 1
        @assert modelinfo.time_link_constr_type == :penalty
        @views begin
            # for convenience
            β = zeros(ngen, T)
            for g=1:ngen
                β[g,:] .= opfdata.generators[g].ramp_agc
            end

            #----------------------------------------------------------------------------
            true_pg_dual = -λ.ramping[:,2:T]
            lagrangian_pg_err = -λprev.ramping[:,2:T]
            penalty_pg_forward_err = -algparams.ρ_t*(
                                        (algparams.jacobi ? xprev.Pg[:,1,1:(T-1)] : x.Pg[:,1,1:(T-1)]) .-
                                        x.Pg[:,1,2:T] .+ x.St[:,2:T] .+ xprev.Zt[:,2:T] .- β[:,2:T]
                                    )
            penalty_pg_reverse_err = +algparams.ρ_t*(
                                        x.Pg[:,1,1:(T-1)] .- xprev.Pg[:,1,2:T] .+ xprev.St[:,2:T] .+ xprev.Zt[:,2:T] .- β[:,2:T]
                                    )
            prox_pg_err = algparams.τ*(x.Pg[:,1,:] .- xprev.Pg[:,1,:])
            #----------------------------------------------------------------------------

            err_pg[:,1,1:(T-1)] += -true_pg_dual .+ lagrangian_pg_err .- penalty_pg_reverse_err
            err_pg[:,1,2:T] += true_pg_dual .- lagrangian_pg_err .- penalty_pg_forward_err
            err_pg[:,1,1:T] -= prox_pg_err[:,1:T]
            err_st[:,2:T] += penalty_pg_forward_err
            err_st[:,2:T] += -true_pg_dual .+ lagrangian_pg_err
            err_zt[:,2:T] += -true_pg_dual .+ lagrangian_pg_err - (algparams.ρ_t*(
                                        x.Pg[:,1,1:(T-1)] .- x.Pg[:,1,2:T] .+ x.St[:,2:T] .+ x.Zt[:,2:T] .- β[:,2:T]
                                    ))
            err_zt[:,2:T] -= (algparams.τ*(x.Zt[:,2:T] - xprev.Zt[:,2:T]))
        end
    end

    if K > 1 && algparams.decompCtgs
        @assert modelinfo.ctgs_link_constr_type ∈ [:frequency_ctrl, :preventive_penalty, :corrective_penalty]
        @views begin
            # for convenience
            β = zeros(ngen, K, T)
            pg_base = zeros(ngen, K, T)
            pg_base_prev = zeros(ngen, K, T)
            if modelinfo.ctgs_link_constr_type ∈ [:corrective_inequality, :corrective_equality, :corrective_penalty]
                for g=1:ngen
                    β[g,:,:] .= opfdata.generators[g].scen_agc
                end
            end
            for k=2:K, g=1:ngen
                pg_base[g,k,:] .= x.Pg[g,1,:]
                pg_base_prev[g,k,:] .= xprev.Pg[g,1,:]
            end

            #----------------------------------------------------------------------------
            true_pg_ctgs_dual = -λ.ctgs[:,2:K,:]
            lagrangian_pg_ctgs_err = -λprev.ctgs[:,2:K,:]
            penalty_pg_base_err = pg_base[:,2:K,:] .-
                                        xprev.Pg[:,2:K,:] .+ xprev.Sk[:,2:K,:] .+ xprev.Zk[:,2:K,:] .- β[:,2:K,:]
            penalty_pg_ctgs_err = (algparams.jacobi ? pg_base_prev[:,2:K,:] : pg_base[:,2:K,:]) -
                                            x.Pg[:,2:K,:] .+     x.Sk[:,2:K,:] .+ xprev.Zk[:,2:K,:] .- β[:,2:K,:]
            prox_pg_err = algparams.τ*(x.Pg[:,2:K,:] .- xprev.Pg[:,2:K,:])
            for g=1:ngen
                penalty_pg_ctgs_err[g,:,:] += opfdata.generators[g].alpha*xprev.ωt[2:K,:]
                penalty_pg_base_err[g,:,:] += opfdata.generators[g].alpha*xprev.ωt[2:K,:]
            end
            penalty_pg_base_err .= algparams.ρ_c*penalty_pg_base_err
            penalty_pg_ctgs_err .= -algparams.ρ_c*penalty_pg_ctgs_err
            #----------------------------------------------------------------------------

            err_pg[:,1,:] += dropdims(sum(-true_pg_ctgs_dual .+ lagrangian_pg_ctgs_err .- penalty_pg_base_err; dims = 2); dims = 2)
            err_pg[:,2:K,:] += true_pg_ctgs_dual .- lagrangian_pg_ctgs_err .- penalty_pg_ctgs_err .- prox_pg_err
            if modelinfo.ctgs_link_constr_type == :frequency_ctrl
                for g=1:ngen
                    err_ωt[2:K,:] += opfdata.generators[g].alpha*(
                                        -true_pg_ctgs_dual[g,:,:].+lagrangian_pg_ctgs_err[g,:,:].-
                                        (algparams.ρ_c*(
                                            pg_base[g,2:K,:] .- x.Pg[g,2:K,:] .+ (opfdata.generators[g].alpha*x.ωt[2:K,:])
                                        ))
                                    )
                    err_ωt[2:K,:] -= algparams.τ*(x.ωt[2:K,:] - xprev.ωt[2:K,:])
                end
            else
                err_sk[:,2:K,:] += penalty_pg_ctgs_err
            end
            if modelinfo.time_link_constr_type ∈ [:preventive_equality, :preventive_penalty, :corrective_equality, :corrective_penalty]
                err_sk[:,2:K,:] += -true_pg_ctgs_dual .+ lagrangian_pg_ctgs_err
            end
            if modelinfo.time_link_constr_type ∈ [:preventive_penalty, :corrective_penalty]
                err_zk[:,2:K,:] += -true_pg_ctgs_dual .+ lagrangian_pg_ctgs_err - (algparams.ρ_c*(
                                            pg_base .- x.Pg[:,2:K,:] .+ x.Sk[:,2:K,:] .+ x.Zk[:,2:K,:] .- β[:,2:K,:]
                                        ))
                err_zk[:,2:K,:] -= (algparams.τ*(x.Zk[:,2:K,:] - xprev.Zk[:,2:K,:]))
            end
        end
    end

    err_pg_view = view(err_pg, :, :, :)
    err_ωt_view = view(err_ωt, :, :)
    err_st_view = view(err_st, :, :)
    err_zt_view = view(err_zt, :, :)
    err_sk_view = view(err_sk, :, :, :)
    err_zk_view = view(err_zk, :, :, :)

    dual_error = CatView(err_pg_view, err_ωt_view, err_st_view, err_zt_view, err_sk_view, err_zk_view)

    return norm(dual_error, lnorm)
end

function opf_block_get_auglag_penalty_expr(
    blk::Int, opfmodel::JuMP.Model,
    modelinfo::ModelParams,
    opfdata::OPFData,
    k, t,
    algparams::AlgParams,
    primal::PrimalSolution,
    dual::DualSolution
)

    (ngen, K, T) = size(primal.Pg)
    gens = opfdata.generators
    @assert t >= 1 && t <= T
    @assert k >= 1 && k <= K
    @assert modelinfo.num_time_periods == 1
    @assert modelinfo.num_ctgs >= 0
    @assert k == 1 || algparams.decompCtgs

    # get variables
    Pg = opfmodel[:Pg]
    St = opfmodel[:St]
    Sk = opfmodel[:Sk]

    auglag_penalty = @expression(opfmodel, 0.5algparams.τ*sum((Pg[g,1,1] - primal.Pg[g,k,t])^2 for g=1:ngen))

    if !algparams.decompCtgs || k == 1
        if t > 1
            ramp_link_expr_prev =
                @expression(opfmodel,
                    [g=1:ngen],
                        primal.Pg[g,1,t-1] - Pg[g,1,1] + St[g,1] + primal.Zt[g,t] - gens[g].ramp_agc
                )
            auglag_penalty += sum(  dual.ramping[g,t]*(+ramp_link_expr_prev[g])   +
                                    0.5*algparams.ρ_t*(+ramp_link_expr_prev[g])^2
                                for g=1:ngen)
        end
        if t < T
            ramp_link_expr_next =
                @expression(opfmodel,
                    [g=1:ngen],
                        Pg[g,1,1] - primal.Pg[g,1,t+1] + primal.St[g,t+1] + primal.Zt[g,t+1] - gens[g].ramp_agc
                )
            auglag_penalty += sum(  dual.ramping[g,t+1]*(+ramp_link_expr_next[g]) +
                                    0.5*algparams.ρ_t*(+ramp_link_expr_next[g])^2
                                for g=1:ngen)
        end
    end


    if algparams.decompCtgs && K > 1
        @assert modelinfo.num_ctgs == 0
        β = [gens[g].scen_agc for g=1:ngen]
        (modelinfo.ctgs_link_constr_type ∉ [:corrective_inequality, :corrective_equality, :corrective_penalty]) &&
            (β .= 0)

        if k > 1
            ctgs_link_expr_prev =
                @expression(opfmodel,
                    [g=1:ngen],
                        primal.Pg[g,1,t] - Pg[g,1,1] + (gens[g].alpha*primal.ωt[k,t]) + Sk[g,1,1] + primal.Zk[g,k,t] - β[g]
                )
            auglag_penalty += sum(    dual.ctgs[g,k,t]*(+ctgs_link_expr_prev[g]) +
                                    0.5*algparams.ρ_c*(+ctgs_link_expr_prev[g])^2
                                for g=1:ngen)
        else
            ctgs_link_expr_next =
                @expression(opfmodel,
                    [g=1:ngen,j=2:K],
                        Pg[g,1,1] - primal.Pg[g,j,t] + (gens[g].alpha*primal.ωt[j,t]) + primal.Sk[g,j,t] + primal.Zk[g,j,t] - β[g]
                )
            auglag_penalty += sum(     dual.ctgs[g,j,t]*(+ctgs_link_expr_next[g,j]) +
                                    0.5*algparams.ρ_c*(+ctgs_link_expr_next[g,j])^2
                                for j=2:K, g=1:ngen)
        end
    end

    drop_zeros!(auglag_penalty)

    return auglag_penalty
end
