
"""
    AbstractBlockModel

Abstract supertype for the definition of block subproblems.
"""
abstract type AbstractBlockModel end

"""
    init!(block::AbstractBlockModel, algparams::AlgParams)

Init the optimization model by creating variables and constraints
inside the model.

"""
function init! end

"""
    optimize!(block::AbstractBlockModel, x0::AbstractArray, algparams::AlgParams)

Solve the optimization problem, starting from an initial
variable `x0`. The optimization solver is specified in
`algparams.optimizer`.

"""
function optimize! end

"""
    get_solution(block::AbstractBlockModel, output)

Return the solution of the optimization as a named tuple `solution`,
with fields

- `status::MOI.TerminationStatus`: final status returned by the solver
- `minimum::Float64`: optimal objective found
- `vm::AbstractArray`: optimal values of voltage magnitudes
- `va::AbstractArray`: optimal values of voltage angles
- `pg::AbstractArray`: optimal values of active power generations
- `qg::AbstractArray`: optimal values of reactive power generations
- `ωt::AbstractArray`: optimal values of frequency
- `st::AbstractArray`: optimal values of slack variables
"""
function get_solution end

# Objective
"""
    set_objective!(
        block::AbstractBlockModel,
        algparams::AlgParams,
        primal::PrimalSolution,
        dual::DualSolution
    )

Update the objective inside `block`'s optimization subproblem.
The new objective updates the coefficients of the penalty
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
"""
    JuMPBlockModel(
        blk::Int,
        opfdata::OPFData, raw_data::RawData,
        modelinfo::ModelParams, t::Int, k::Int, T::Int,
    )
)

Block model using the modeler JuMP to define the optimal power flow
problem. Used inside `OPFBlocks`, for decomposition purpose.

# Arguments

- `blk::Int`: ID of the block represented by this model
- `opfdata::OPFData`: data used to build the optimal power flow problem.
- `raw_data::RawData`: same data, in raw format
- `modelinfo::ModelParams`: parameters related to specification of the optimization model
- `t::Int`: current time-step. Value should be between `1` and `T`.
- `k::Int`: current contingency
- `T::Int`: final horizon

"""
struct JuMPBlockModel <: AbstractBlockModel
    id::Int
    k::Int
    t::Int
    model::JuMP.Model
    data::OPFData
    params::ModelParams
end

function JuMPBlockModel(
    blk::Int,
    opfdata::OPFData, raw_data::RawData,
    modelinfo::ModelParams, t::Int, k::Int, T::Int,
)
    model = JuMP.Model()
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
        add_ctgs_linking_constraints!(block, algparams)
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
        opf_model_add_real_power_balance_constraints(opfmodel, opfdata_c, opfmodel[:Pg][:,j,1], opfdata_c.Pd[:,1], opfmodel[:Vm][:,j,1], opfmodel[:Va][:,j,1], σ_re)
        opf_model_add_imag_power_balance_constraints(opfmodel, opfdata_c, opfmodel[:Qg][:,j,1], opfdata_c.Qd[:,1], opfmodel[:Vm][:,j,1], opfmodel[:Va][:,j,1], σ_im)
        opf_model_add_line_power_constraints(opfmodel, opfdata_c, opfmodel[:Vm][:,j,1], opfmodel[:Va][:,j,1], σ_fr, σ_to)
    end
    return opfmodel
end

function set_objective!(block::JuMPBlockModel, algparams::AlgParams,
                        primal::PrimalSolution, dual::DualSolution)
    blk = block.id
    opfmodel = block.model
    opfdata = block.data
    modelinfo = block.params
    k, t = block.k, block.t

    obj_expr = compute_objective_function(opfmodel, opfdata, modelinfo)
    auglag_penalty = opf_block_get_auglag_penalty_expr(
        blk, opfmodel, modelinfo, opfdata, k, t, algparams, primal, dual)
    @objective(opfmodel, Min, obj_expr + auglag_penalty)
    return
end

function get_solution(block::JuMPBlockModel)
    opfmodel = block.model
    blk = block.id
    status = termination_status(opfmodel)
    if status ∉ MOI_OPTIMAL_STATUSES
        @warn("Block $blk subproblem not solved to optimality. status: $status")
    end
    if !has_values(opfmodel)
        error("no solution vector available in block $blk subproblem")
    end

    solution = (
        status=status,
        minimum=JuMP.objective_value(opfmodel),
        vm=JuMP.value.(opfmodel[:Vm]),
        va=JuMP.value.(opfmodel[:Va]),
        pg=JuMP.value.(opfmodel[:Pg]),
        qg=JuMP.value.(opfmodel[:Qg]),
        ωt=JuMP.value.(opfmodel[:ωt]),
        st=JuMP.value.(opfmodel[:St]),
    )
    return solution
end

function set_start_values!(block::JuMPBlockModel, x0)
    JuMP.set_start_value.(all_variables(block.model), x0)
end

function optimize!(block::JuMPBlockModel, x0::AbstractArray, algparams::AlgParams)
    blk = block.id
    opfmodel = block.model
    set_start_values!(block, x0)
    JuMP.optimize!(block.model)
    return get_solution(block)
end

function add_variables!(block::JuMPBlockModel, algparams::AlgParams)
    opf_model_add_variables(
        block.model, block.data, block.params, algparams,
    )
end

function add_ctgs_linking_constraints!(block::JuMPBlockModel, algparams)
    opf_model_add_ctgs_linking_constraints(
        block.model, block.data, block.params,
    )
end

### Implementation of ExaBlockModel
"""
    ExaBlockModel(
        blk::Int,
        opfdata::OPFData, raw_data::RawData,
        modelinfo::ModelParams, t::Int, k::Int, T::Int,
    )
)

Block model using the package ExaPF to define the optimal power flow
problem. Used inside `OPFBlocks`, for decomposition purpose.

# Arguments

- `blk::Int`: ID of the block represented by this model
- `opfdata::OPFData`: data used to build the optimal power flow problem.
- `raw_data::RawData`: same data, in raw format
- `modelinfo::ModelParams`: parameters related to specification of the optimization model
- `t::Int`: current time-step. Value should be between `1` and `T`.
- `k::Int`: current contingency
- `T::Int`: final horizon

"""
struct ExaBlockModel <: AbstractBlockModel
    id::Int
    k::Int
    t::Int
    model::ExaPF.AbstractNLPEvaluator
    data::OPFData
    params::ModelParams
end

function ExaBlockModel(
    blk::Int,
    opfdata::OPFData, raw_data::RawData,
    modelinfo::ModelParams, t::Int, k::Int, T::Int;
    device::TargetDevice=CPU, nr_tol=1e-10,
)

    data = Dict{String, Array}()
    data["bus"] = raw_data.bus_arr
    data["branch"] = raw_data.branch_arr
    data["gen"] = raw_data.gen_arr
    data["cost"] = raw_data.costgen_arr
    data["baseMVA"] = [raw_data.baseMVA]

    power_network = PS.PowerNetwork(data)

    if t == 1
        time = ExaPF.Origin
    elseif t == T
        time = ExaPF.Final
    else
        time = ExaPF.Normal
    end

    # Instantiate model in memory
    if device == CPU
        target = ExaPF.CPU()
    elseif device == CUDADevice
        target = ExaPF.CUDADevice()
    else
        error("Device $(device) is not supported by ExaPF")
    end
    model = ExaPF.ProxALEvaluator(power_network, time;
                                  device=target, ε_tol=nr_tol)
    return ExaBlockModel(blk, k, t, model, opfdata, modelinfo)
end

function init!(block::ExaBlockModel, algparams::AlgParams)
    opfmodel = block.model
    baseMVA = block.data.baseMVA

    # Reset optimizer
    ExaPF.reset!(opfmodel)
    # Get params
    opfdata = block.data
    modelinfo = block.params
    Kblock = modelinfo.num_ctgs + 1
    t, k = block.t, block.k
    # Generators
    gens = block.data.generators
    ramp_agc = [g.ramp_agc for g in gens]

    # Sanity check
    @assert modelinfo.num_time_periods == 1
    @assert !algparams.decompCtgs || Kblock == 1

    # TODO: currently, only one contingency is supported
    j = 1
    pd = opfdata.Pd[:,1] / baseMVA
    qd = opfdata.Qd[:,1] / baseMVA
    ExaPF.setvalues!(opfmodel, PS.ActiveLoad(), pd)
    ExaPF.setvalues!(opfmodel, PS.ReactiveLoad(), qd)
    # Set bounds on slack variables s
    copyto!(opfmodel.s_max, 2 .* ramp_agc)
    opfmodel.scale_objective = modelinfo.obj_scale

    return opfmodel
end

function add_ctgs_linking_constraints!(block::ExaBlockModel)
    error("Contingencies are not supported in ExaPF")
end

function update_penalty!(block::ExaBlockModel, algparams::AlgParams,
                         primal::PrimalSolution, dual::DualSolution)
    examodel = block.model
    opfdata = block.data
    modelinfo = block.params
    # Generators
    gens = block.data.generators

    t, k = block.t, block.k
    ramp_agc = [g.ramp_agc for g in gens]

    time = examodel.time

    # Update current values
    λf = dual.ramping[:, t]
    pgc = primal.Pg[:, k, t]
    ExaPF.update_multipliers!(examodel, ExaPF.Current(), λf)
    ExaPF.update_primal!(examodel, ExaPF.Current(), pgc)
    # Update parameters
    examodel.τ = algparams.τ

    # Update previous values
    if time != ExaPF.Origin
        pgf = primal.Pg[:, 1, t-1] .+ primal.Zt[:, t] .- ramp_agc
        ExaPF.update_primal!(examodel, ExaPF.Previous(), pgf)
        # Update parameters
        examodel.ρf = algparams.ρ_t[1, t]
    end

    # Update next values
    if time != ExaPF.Final
        λt = dual.ramping[:, t+1]
        pgt = primal.Pg[:, 1, t+1] .- primal.St[:, t+1] .- primal.Zt[:, t+1] .+ ramp_agc
        ExaPF.update_multipliers!(examodel, ExaPF.Next(), λt)
        ExaPF.update_primal!(examodel, ExaPF.Next(), pgt)
        # Update parameters
        examodel.ρt = algparams.ρ_t[1, t+1]
    end
end

function set_objective!(block::ExaBlockModel, algparams::AlgParams,
                        primal::PrimalSolution, dual::DualSolution)
    update_penalty!(block, algparams, primal, dual)
    return
end

function get_solution(block::ExaBlockModel, output)
    opfmodel = block.model

    # Check optimization status
    status = output.status
    if status ∉ MOI_OPTIMAL_STATUSES
        @warn("Block $(block.id) subproblem not solved to optimality. status: $status")
    end

    # Get optimal solution in reduced space
    nu = opfmodel.nu
    x♯ = output.minimizer
    u♯ = x♯[1:nu]
    s♯ = x♯[nu+1:end]
    # Unroll solution in full space
    ## i) Project in null-space of equality constraints
    ExaPF.update!(opfmodel, x♯)
    pg = get(opfmodel, PS.ActivePower())
    qg = get(opfmodel, PS.ReactivePower())
    vm = get(opfmodel, PS.VoltageMagnitude())
    va = get(opfmodel, PS.VoltageAngle())

    solution = (
        status=status,
        minimum=output.minimum,
        vm=vm,
        va=va,
        pg=pg,
        qg=qg,
        ωt=[0.0], # At the moment, no frequency variable in ExaPF
        st=s♯,
    )
    return solution
end

function set_start_values!(block::ExaBlockModel, x0)
    ngen = length(block.data.generators)
    nbus = length(block.data.buses)
    # Only one contingency, at the moment
    K = 1
    # Extract values from array x0
    pg = x0[1:ngen*K]
    qg = x0[ngen*K+1:2*ngen*K]
    vm = x0[2*ngen*K+1:2*ngen*K+nbus*K]
    va = x0[2*ngen*K+nbus*K+1:2*ngen*K+2*nbus*K]
    # Transfer them to ExaPF. Initial control will be automatically updated
    ExaPF.transfer!(block.model, vm, va, pg, qg)
end

function optimize!(block::ExaBlockModel, x0::AbstractArray, algparams::AlgParams)
    blk = block.id
    opfmodel = block.model
    optimizer = algparams.gpu_optimizer

    if isa(optimizer, MOI.OptimizerWithAttributes)
        optimizer = MOI.instantiate(optimizer)
    end

    # Optimize with optimizer, using ExaPF model
    output = ExaPF.optimize!(optimizer, opfmodel)
    # Recover solution
    solution = get_solution(block, output)

    if isa(optimizer, MOI.OptimizerWithAttributes)
        MOI.empty!(optimizer)
    end

    return solution
end

