"""
    OPFBlockData

Specifies individual NLP block data.

- `blkCount::Int64`: number of decomposed NLP blocks
- `blkIndex::CartesianIndices`: each element is a tuple (k,t) where `k` is the contingency and `t` is the time period of that NLP block
- `blkModel::Vector{JuMP.Model}`: the JuMP representation of each NLP block
- `blkOpfdt::Vector{OPFData}`: the ACOPF instance data corresponding to each  NLP block
- `blkMInfo::Vector{ModelParams}`: the [Model parameters](@ref) corresponding to each NLP block
- `colCount::Int64`: the number of decision variables in each NLP block (must be equal across blocks)
- `colValue::Array{Float64,2}`: `colCount` ``\\times`` `blkCount` array representing the solution vector 
"""
mutable struct OPFBlockData
    blkCount::Int64
    blkIndex::CartesianIndices
    blkModel::Vector{JuMP.Model}
    blkOpfdt::Vector{OPFData}
    blkMInfo::Vector{ModelParams}
    colCount::Int64
    colValue::Array{Float64,2}

    function OPFBlockData(opfdata::OPFData, rawdata::RawData,
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


        new(blkCount,blkIndex,blkModel,blkOpfdt,blkMInfo,colCount,colValue)
    end
end

"""
    opf_block_model_initialize(blk::Int
                               opfblocks::OPFBlockData,
                               rawdata::RawData,
                               algparams::AlgParams)

Initializes the JuMP model representation of
NLP block `blk` (this is guaranteed to be in the range
`[1, opfblocks.blkCount]`) using the instance data
`raw` and algorithm parameters `algparams`.
The contingency number `k` and  time period `t` of the
NLP block can be extracted as follows:
```julia
k = opfblocks.blkIndex[blk][1]
t = opfblocks.blkIndex[blk][2]
```
"""
function opf_block_model_initialize(blk::Int, opfblocks::OPFBlockData, rawdata::RawData,
                                    algparams::AlgParams)
    @assert blk >= 1 && blk <= opfblocks.blkCount

    opfmodel = JuMP.Model(algparams.optimizer)
    JuMP.set_optimizer_attribute(opfmodel, "max_iter", algparams.nlpiterlim)
    opfdata = opfblocks.blkOpfdt[blk]
    modelinfo = opfblocks.blkMInfo[blk]
    Kblock = modelinfo.num_ctgs + 1

    opf_model_add_variables(opfmodel, opfdata, modelinfo, algparams)
    if !algparams.decompCtgs
        opf_model_add_ctgs_linking_constraints(opfmodel, opfdata, modelinfo)
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

"""
    opf_block_set_objective(blk::Int,
                            opfmodel::JuMP.Model,
                            opfblocks::OPFBlockData,
                            algparams::AlgParams,
                            primal::PrimalSolution,
                            dual::DualSolution)

Sets the  objective function in `opfmodel` of the
NLP block `blk` using the algorithm parameters `algparams`,
and corresponding to a reference primal solution `primal`
and reference dual solution `dual`. 
The objective function is the sum of the single-period ACOPF
corresponding to NLP block `blk`'s time period and the
expression returned by `opf_block_get_auglag_penalty_expr`.
"""
function opf_block_set_objective(blk::Int, opfmodel::JuMP.Model, opfblocks::OPFBlockData,
                                 algparams::AlgParams,
                                 primal::PrimalSolution,
                                 dual::DualSolution)
    @assert blk >= 1 && blk <= opfblocks.blkCount

    opfdata = opfblocks.blkOpfdt[blk]
    modelinfo = opfblocks.blkMInfo[blk]
    obj_expr = compute_objective_function(opfmodel, opfdata, modelinfo)
    auglag_penalty = opf_block_get_auglag_penalty_expr(blk, opfmodel, opfblocks, algparams, primal, dual)
    @objective(opfmodel, Min, obj_expr + auglag_penalty)
    return nothing
end

"""
    opf_block_get_auglag_penalty_expr(blk::Int,
                                      opfmodel::JuMP.Model,
                                      opfblocks::OPFBlockData,
                                      algparams::AlgParams,
                                      primal::PrimalSolution,
                                      dual::DualSolution)

Let `k` and `t` denote the contingency number and time period of
the NLP block `blk` that can be extracted as follows:
```julia
k = opfblocks.blkIndex[blk][1]
t = opfblocks.blkIndex[blk][2]
```
Then, depending on the values of `k`, `t`, `algparams.decompCtgs` and
`opfblocks.blkMInfo[blk]`, this function must return an appropriate expression.

Consider first the case where `algparams.decompCtgs == false`.
In the following expressions, we use ``\\mathbb{I}[...]`` to denote the indicator function.
Also, unless otherwise indicated, 
- the values of ``z_g`` are always parameters in this expression and must be taken from `primal.Zt`
- the values of ``p_g`` and ``s_g`` variables that are not indexed with `t` are parameters in this expression and must be taken from `primal.Pg` and `primal.St`
- the values of ``\\lambda``, ``\\lambda^p``, and ``\\lambda^n`` (without contingency subscript) must be taken from `dual.ramping`, `dual.ramping_p` and `dual.ramping_n`, respectively
- the values of ``\\rho`` (without contingency subscript) must be taken from `algparams.ρ_t`
- the values of ``\\tau`` must be taken from `algparams.τ`


We consider the following cases.

* If `opfblocks.blkMInfo[blk] == :inequality`, then this function must return:
```math
\\begin{aligned}
\\sum_{g \\in G} \\Bigg\\{
& 0.5\\tau [p^0_{g,t} - \\mathrm{primal}.p^0_{g,t}]^2 \\\\
&+\\mathbb{I}[t > 1]\\Big(
\\lambda^p_{gt}[p^0_{g,t-1} - p^0_{g,t} - r_g] -
\\lambda^n_{gt}[p^0_{g,t-1} - p^0_{g,t} - r_g] +
0.5\\rho_{gt}[p^0_{g,t-1} - p^0_{g,t} + s_{g,t} - r_g]^2
\\Big) \\\\
&+\\mathbb{I}[t < T]\\Big(
\\lambda^p_{g,t+1}[p^0_{g,t} - p^0_{g,t+1} - r_g] -
\\lambda^n_{g,t+1}[p^0_{g,t} - p^0_{g,t+1} - r_g] +
0.5\\rho_{g,t+1}[p^0_{g,t} - p^0_{g,t+1} + s_{g,t+1} - r_g]^2
\\Big) \\Bigg\\}
\\end{aligned}
```

* If `opfblocks.blkMInfo[blk] == :equality`, then this function must return:
```math
\\begin{aligned}
\\sum_{g \\in G} \\Bigg\\{
& 0.5\\tau [p^0_{g,t} - \\mathrm{primal}.p^0_{g,t}]^2 \\\\
&+\\mathbb{I}[t > 1]\\Big(
\\lambda_{gt}[p^0_{g,t-1} - p^0_{g,t} + s_{g,t} - r_g] +
0.5\\rho_{gt}[p^0_{g,t-1} - p^0_{g,t} + s_{g,t} - r_g]^2
\\Big) \\\\
&+\\mathbb{I}[t < T]\\Big(
\\lambda_{g,t+1}[p^0_{g,t} - p^0_{g,t+1} + s_{g,t+1} - r_g] +
0.5\\rho_{g,t+1}[p^0_{g,t} - p^0_{g,t+1} + s_{g,t+1} - r_g]^2
\\Big) \\Bigg\\}
\\end{aligned}
```

* If `opfblocks.blkMInfo[blk] == :penalty`, then this function must return:
```math
\\begin{aligned}
\\sum_{g \\in G} \\Bigg\\{
& 0.5\\tau [p^0_{g,t} - \\mathrm{primal}.p^0_{g,t}]^2 \\\\
&+\\mathbb{I}[t > 1]\\Big(
\\lambda_{gt}[p^0_{g,t-1} - p^0_{g,t} + s_{g,t} + z_{g,t} - r_g] +
0.5\\rho_{gt}[p^0_{g,t-1} - p^0_{g,t} + s_{g,t} + z_{g,t} - r_g]^2
\\Big) \\\\
&+\\mathbb{I}[t < T]\\Big(
\\lambda_{g,t+1}[p^0_{g,t} - p^0_{g,t+1} + s_{g,t+1} + z_{g,t+1} - r_g] +
0.5\\rho_{g,t+1}[p^0_{g,t} - p^0_{g,t+1} + s_{g,t+1} + z_{g,t+1} - r_g]^2
\\Big) \\Bigg\\}
\\end{aligned}
```
"""
function opf_block_get_auglag_penalty_expr(blk::Int, opfmodel::JuMP.Model, opfblocks::OPFBlockData,
                                           algparams::AlgParams,
                                           primal::PrimalSolution,
                                           dual::DualSolution)
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

"""
    opf_block_solve_model(blk::Int,
                          opfmodel::JuMP.Model,
                          opfblocks::OPFBlockData)

Solve the NLP block `blk`'s JuMP model `opfmodel` and 
return the (locally optimal) solution vector.
The model will be initialized using `opfblocks.colValue[:,blk]`.
"""
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

