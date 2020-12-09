
abstract type AbstractBlockModel end

"""
    init!(block::AbstractBlockModel, algparams::AlgParams, indexes)

Feed the optimization model by creating variables and constraints
inside the model.

"""
function init! end

"""
    optimize!(block::AbstractBlockModel, x0)

Solve the optimization problem, starting from an initial
variable `x0`.

"""
function optimize! end

# Objective
"""
    set_objective!(
        block::AbstractBlockModel,
        algparams::AlgParams,
        primal::PrimalSolution,
        dual::DualSolution
    )

Update the objective inside `block`'s optimization problem.
The new objective update the coefficients of the penalty
terms, to reflect the new `primal` and `dual` solutions
passed in the arguments.

"""
function set_objective! end

# Variables
"""
    add_variables!(block::AbstractBlockModel, algparams::AlgParams)

Add all optimization variables into the decomposed optimization
model `block`.

"""
function add_variables! end

# Constraints
function add_ctgs_linking_constraints! end

### Implementation of JuMPBlockModel
struct JuMPBlockModel <: AbstractBlockModel
    id::Int
    k::Int
    t::Int
    model::JuMP.Model
    data::OPFData
    params::ModelParams
end

function JuMPBlockModel(blk, opfdata, modelinfo, indexes)
    model = JuMP.Model()
    k = indexes[block.id][1]
    t = indexes[block.id][2]
    return JuMPBlockModel(blk, k, t, model, opfdata, modelinfo)
end

function init!(block::JuMPBlockModel, algparams::AlgParams)
    opfmodel = block.model
    # Reset optimizer
    Base.empty!(opfmodel)

    # Pass optimizer to model
    JuMP.set_optimizer(opfmodel, algparams.optimizer)
    JuMP.set_optimizer_attribute(opfmodel, "max_iter", algparams.nlpiterlim)

    # Get params
    opfdata = block.data
    modelinfo = block.params
    Kblock = modelinfo.num_ctgs + 1

    # Sanity check
    @assert modelinfo.num_time_periods == 1
    @assert !algparams.decompCtgs || Kblock == 1

    add_variables!(block, algparams)
    if !algparams.decompCtgs
        add_ctgs_linking_constraints!(block)
    end

    t, k = block.t, block.k

    (t == 1) &&
        fix.(opfmodel[:St][:,1], 0; force = true)
    (k == 1 && algparams.decompCtgs) &&
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
        σ_re = opfmodel[:sigma_real][:,j,1]
        σ_im = opfmodel[:sigma_imag][:,j,1]
        σ_fr = opfmodel[:sigma_lineFr][:,j,1]
        σ_to = opfmodel[:sigma_lineTo][:,j,1]
    else
        zb = zeros(length(opfdata.buses))
        zl = zeros(length(opfdata.lines))
        σ_re = zb
        σ_im = zb
        σ_fr = zl
        σ_to = zl
    end

    @views for j=1:Kblock
        opfdata_c = (j == 1) ? opfdata :
            opf_loaddata(rawdata; lineOff = opfdata.lines[rawdata.ctgs_arr[j - 1]], time_horizon_start = t, time_horizon_end = t, load_scale = modelinfo.load_scale, ramp_scale = modelinfo.ramp_scale)
        add_real_power_balance_constraints!(opfmodel, opfdata_c, opfmodel[:Pg][:,j,1], opfdata_c.Pd[:,1], opfmodel[:Vm][:,j,1], opfmodel[:Va][:,j,1], σ_re)
        add_imag_power_balance_constraints!(opfmodel, opfdata_c, opfmodel[:Qg][:,j,1], opfdata_c.Qd[:,1], opfmodel[:Vm][:,j,1], opfmodel[:Va][:,j,1], σ_im)
        add_line_power_constraints!(opfmodel, opfdata_c, opfmodel[:Vm][:,j,1], opfmodel[:Va][:,j,1], σ_fr, σ_to)
    end
    return opfmodel
end

function set_objective!(block::JuMPBlockModel, algparams::AlgParams,
                        primal::PrimalSolution, dual::DualSolution)
    opfmodel = block.model
    opfdata = block.data
    modelinfo = block.params

    obj_expr = objective_function(opfmodel, opfdata, modelinfo)
    auglag_penalty = opf_block_get_auglag_penalty_expr(blk, opfmodel, opfblocks, algparams, primal, dual)
    @objective(opfmodel, Min, obj_expr + auglag_penalty)
    return
end

function optimize!(block::JuMPBlockModel, x0)
    blk = block.id
    opfmodel = block.model
    set_start_value.(all_variables(opfmodel), x0)
    optimize!(opfmodel)

    status = termination_status(opfmodel)
    if status ∉ MOI_OPTIMAL_STATUSES
        @warn("Block $blk subproblem not solved to optimality. status: $status")
    end
    if !has_values(opfmodel)
        error("no solution vector available in block $blk subproblem")
    end
    solution = value.(all_variables(opfmodel))
    return solution
end

function add_variables!(block::JuMPBlockModel, algparams::AlgParams)
    opf_model_add_variables(
        block.model, block.data, block.params, algparams,
    )
end

function add_ctgs_linking_constraints!(block::JuMPBlockModel)
    opf_model_add_ctgs_linking_constraints(
        block.model, block.data, block.params, algparams,
    )
end

