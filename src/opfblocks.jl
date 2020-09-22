#
# Algorithm data
#
mutable struct OPFBlockData
    blkCount::Int64
    blkIndex::CartesianIndices
    blkModel::Vector{JuMP.Model}
    blkOptimizer
    blkOpfdt::Vector{OPFData}
    blkMInfo::Vector{ModelParams}
    colCount::Int64
    colValue::Array{Float64,2}

    function OPFBlockData(opfdata::OPFData, rawdata::RawData, optimizer;
                          modelinfo::ModelParams = ModelParams(),
                          algparams::AlgParams = AlgParams())
        ngen  = length(opfdata.generators)
        nbus  = length(opfdata.buses)
        nline = length(opfdata.lines)
        T = modelinfo.num_time_periods
        K = modelinfo.num_ctgs + 1 # base case counted separately


        blkCount = algparams.decompCtgs ? (T*K) : T
        blkIndex = algparams.decompCtgs ? CartesianIndices((1:K,1:T)) : CartesianIndices((1:1,1:T))
        colCount = ((algparams.decompCtgs ? 1 : K)*(
                        2nbus + 4ngen + 1 +
                            (Int(modelinfo.allow_constr_infeas)*(2nbus + 2nline))
                    )) + 2ngen
        colValue = zeros(colCount,blkCount)


        blkModel = Vector{JuMP.Model}(undef, blkCount)
        blkOpfdt = Vector{OPFData}(undef, blkCount)
        blkMInfo = Vector{ModelParams}(undef, blkCount)
        for blk in LinearIndices(blkIndex)
            k = blkIndex[blk][1]
            t = blkIndex[blk][2]

            blkMInfo[blk] = deepcopy(modelinfo)
            blkMInfo[blk].num_time_periods = 1
            blkOpfdt[blk]= opf_loaddata(rawdata;
                                        time_horizon_start = t,
                                        time_horizon_end = t,
                                        load_scale = modelinfo.load_scale,
                                        ramp_scale = modelinfo.ramp_scale)
            if algparams.decompCtgs
                @assert k > 0
                blkMInfo[blk].num_ctgs = 0
                if k > 1
                    blkOpfdt[blk]= opf_loaddata(rawdata;
                                                time_horizon_start = t,
                                                time_horizon_end = t,
                                                load_scale = modelinfo.load_scale,
                                                ramp_scale = modelinfo.ramp_scale,
                                                lineOff = opfdata.lines[rawdata.ctgs_arr[k - 1]])
                end
            end
        end


        new(blkCount,blkIndex,blkModel,optimizer,blkOpfdt,blkMInfo,colCount,colValue)
    end
end

function opf_block_model_initialize(blk::Int, opfblocks::OPFBlockData, rawdata::RawData;
                                    algparams::AlgParams)
    @assert blk >= 1 && blk <= opfblocks.blkCount

    opfmodel = JuMP.Model(opfblocks.blkOptimizer)
    opfdata = opfblocks.blkOpfdt[blk]
    modelinfo = opfblocks.blkMInfo[blk]
    Kblock = modelinfo.num_ctgs + 1

    opf_model_add_variables(opfmodel, opfdata; modelinfo = modelinfo, algparams = algparams)
    if !algparams.decompCtgs
        opf_model_add_ctgs_linking_constraints(opfmodel, opfdata; modelinfo = modelinfo)
    end

    @assert opfblocks.colCount == num_variables(opfmodel)
    @assert modelinfo.num_time_periods == 1
    @assert !algparams.decompCtgs || Kblock == 1
    
    k = opfblocks.blkIndex[blk][1]
    t = opfblocks.blkIndex[blk][2]
    (t==1) &&
        fix.(opfmodel[:St][:,1], 0; force = true)
    (k==1 && algparams.decompCtgs) &&
        fix.(opfmodel[:Sk][:,1,:], 0; force = true)    

    # Fix penalty vars to 0
    fix.(opfmodel[:Zt], 0; force = true)
    if algparams.decompCtgs
        fix.(opfmodel[:Zk], 0; force = true)
        if modelinfo.ctgs_link_constr_type == :frequency_ctrl
            fix.(opfmodel[:ωt], 0; force = true)
        end
    end

    # Add block constraints
    if modelinfo.allow_constr_infeas
        @views for j=1:Kblock
            opfdata_c = (j == 1) ? opfdata : 
                opf_loaddata(rawdata; lineOff = opfdata.lines[rawdata.ctgs_arr[j - 1]], time_horizon_start = t, time_horizon_end = t, load_scale = modelinfo.load_scale, ramp_scale = modelinfo.ramp_scale)
            opf_model_add_real_power_balance_constraints(opfmodel, opfdata_c, opfmodel[:Pg][:,j,1], opfdata_c.Pd[:,1], opfmodel[:Vm][:,j,1], opfmodel[:Va][:,j,1], opfmodel[:sigma_real][:,j,1])
            opf_model_add_imag_power_balance_constraints(opfmodel, opfdata_c, opfmodel[:Qg][:,j,1], opfdata_c.Qd[:,1], opfmodel[:Vm][:,j,1], opfmodel[:Va][:,j,1], opfmodel[:sigma_imag][:,j,1])
            opf_model_add_line_power_constraints(opfmodel, opfdata_c, opfmodel[:Vm][:,j,1], opfmodel[:Va][:,j,1], opfmodel[:sigma_lineFr][:,j,1], opfmodel[:sigma_lineTo][:,j,1])
        end
    else
        zb = zeros(length(opfdata.buses))
        zl = zeros(length(opfdata.lines))
        @views for j=1:Kblock
            opfdata_c = (j == 1) ? opfdata : 
                opf_loaddata(rawdata; lineOff = opfdata.lines[rawdata.ctgs_arr[j - 1]], time_horizon_start = t, time_horizon_end = t, load_scale = modelinfo.load_scale, ramp_scale = modelinfo.ramp_scale)
            opf_model_add_real_power_balance_constraints(opfmodel, opfdata_c, opfmodel[:Pg][:,j,1], opfdata_c.Pd[:,1], opfmodel[:Vm][:,j,1], opfmodel[:Va][:,j,1], zb)
            opf_model_add_imag_power_balance_constraints(opfmodel, opfdata_c, opfmodel[:Qg][:,j,1], opfdata_c.Qd[:,1], opfmodel[:Vm][:,j,1], opfmodel[:Va][:,j,1], zb)
            opf_model_add_line_power_constraints(opfmodel, opfdata_c, opfmodel[:Vm][:,j,1], opfmodel[:Va][:,j,1], zl, zl)
        end
    end
    return opfmodel
end

function opf_block_set_objective(blk::Int, opfmodel::JuMP.Model, opfblocks::OPFBlockData;
                                 primal::PrimalSolution,
                                 dual::DualSolution,
                                 algparams::AlgParams)
    @assert blk >= 1 && blk <= opfblocks.blkCount

    opfdata = opfblocks.blkOpfdt[blk]
    modelinfo = opfblocks.blkMInfo[blk]
    obj_expr = compute_objective_function(opfmodel, opfdata; modelinfo = modelinfo)
    auglag_penalty = opf_block_get_auglag_penalty_expr(blk, opfmodel, opfblocks; algparams = algparams, primal = primal, dual = dual)
    @objective(opfmodel, Min, obj_expr + auglag_penalty)
    return nothing
end

function opf_block_get_auglag_penalty_expr(blk::Int, opfmodel::JuMP.Model, opfblocks::OPFBlockData;
                                           primal::PrimalSolution,
                                           dual::DualSolution,
                                           algparams::AlgParams)
    modelinfo = opfblocks.blkMInfo[blk]
    gens = opfblocks.blkOpfdt[blk].generators
    k = opfblocks.blkIndex[blk][1]
    t = opfblocks.blkIndex[blk][2]

    (ngen, K, T) = size(primal.Pg)
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
            if modelinfo.time_link_constr_type == :inequality
                @assert abs(norm(primal.Zt)) <= algparams.zero
                auglag_penalty += sum(dual.ramping_p[g,t]*(+ramp_link_expr_prev[g] - St[g,1]) +
                                      dual.ramping_n[g,t]*(-ramp_link_expr_prev[g] + St[g,1] - 2gens[g].ramp_agc) +
                                      0.5*algparams.ρ_t[g,t]*(+ramp_link_expr_prev[g])^2
                                    for g=1:ngen)
            else
                auglag_penalty += sum(  dual.ramping[g,t]*(+ramp_link_expr_prev[g])   +
                                      0.5*algparams.ρ_t[g,t]*(+ramp_link_expr_prev[g])^2
                                    for g=1:ngen)
            end
        end
        if t < T
            ramp_link_expr_next = 
                @expression(opfmodel,
                    [g=1:ngen],
                        Pg[g,1,1] - primal.Pg[g,1,t+1] + primal.St[g,t+1] + primal.Zt[g,t+1] - gens[g].ramp_agc
                )
            if modelinfo.time_link_constr_type == :inequality
                @assert abs(norm(primal.Zt)) <= algparams.zero
                auglag_penalty += sum(dual.ramping_p[g,t+1]*(+ramp_link_expr_next[g] - primal.St[g,t+1]) +
                                      dual.ramping_n[g,t+1]*(-ramp_link_expr_next[g] + primal.St[g,t+1] - 2gens[g].ramp_agc) +
                                      0.5*algparams.ρ_t[g,t+1]*(+ramp_link_expr_next[g])^2
                                    for g=1:ngen)
            else
                auglag_penalty += sum(  dual.ramping[g,t+1]*(+ramp_link_expr_next[g]) +
                                      0.5*algparams.ρ_t[g,t+1]*(+ramp_link_expr_next[g])^2
                                    for g=1:ngen)
            end
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
            if modelinfo.ctgs_link_constr_type == :corrective_inequality
                @assert abs(norm(primal.ωt)) <= algparams.zero
                @assert abs(norm(primal.Zk)) <= algparams.zero
                auglag_penalty += sum(  dual.ctgs_p[g,k,t]*(+ctgs_link_expr_prev[g] - Sk[g,1,1]) +
                                        dual.ctgs_n[g,k,t]*(-ctgs_link_expr_prev[g] + Sk[g,1,1] - 2β[g]) +
                                     0.5*algparams.ρ_c[g,k,t]*(+ctgs_link_expr_prev[g])^2
                                    for g=1:ngen)
            else
                auglag_penalty += sum(    dual.ctgs[g,k,t]*(+ctgs_link_expr_prev[g]) +
                                     0.5*algparams.ρ_c[g,k,t]*(+ctgs_link_expr_prev[g])^2
                                    for g=1:ngen)
            end
        else
            ctgs_link_expr_next =
                @expression(opfmodel,
                    [g=1:ngen,j=2:K],
                        Pg[g,1,1] - primal.Pg[g,j,t] + (gens[g].alpha*primal.ωt[j,t]) + primal.Sk[g,j,t] + primal.Zk[g,j,t] - β[g]
                )
            if modelinfo.ctgs_link_constr_type == :corrective_inequality
                @assert abs(norm(primal.ωt)) <= algparams.zero
                @assert abs(norm(primal.Zk)) <= algparams.zero
                auglag_penalty += sum(   dual.ctgs_p[g,j,t]*(+ctgs_link_expr_next[g,j] - primal.Sk[g,j,t]) +
                                         dual.ctgs_n[g,j,t]*(-ctgs_link_expr_next[g,j] + primal.Sk[g,j,t] - 2β[g]) +
                                      0.5*algparams.ρ_c[g,j,t]*(+ctgs_link_expr_next[g,j])^2
                                    for j=2:K, g=1:ngen)
            else
                auglag_penalty += sum(     dual.ctgs[g,j,t]*(+ctgs_link_expr_next[g,j]) +
                                      0.5*algparams.ρ_c[g,j,t]*(+ctgs_link_expr_next[g,j])^2
                                    for j=2:K, g=1:ngen)
            end
        end
    end

    drop_zeros!(auglag_penalty)

    return auglag_penalty
end

function opf_block_solve_model(blk::Int, opfmodel::JuMP.Model, opfblocks::OPFBlockData)
    @assert blk >= 1 && blk <= opfblocks.blkCount
    set_start_value.(all_variables(opfmodel), opfblocks.colValue[:,blk])
    optimize!(opfmodel)
    status = termination_status(opfmodel)
    if status ∉ [MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED]
        println("warning: block $blk subproblem not solved to optimality. status: $status")
    end
    if !has_values(opfmodel)
        error("no solution vector available in block $blk subproblem")
    end
    return value.(all_variables(opfmodel))
end

