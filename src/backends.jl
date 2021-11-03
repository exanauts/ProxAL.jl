"""
    ExaPFBackend <: AbstractBackend

Solve OPF in reduced-space with ExaPF.
"""
struct ExaPFBackend <: AbstractBackend end

"""
    JuMPBackend <: AbstractBackend

Solve OPF in full-space with JuMP/MOI.
"""
struct JuMPBackend <: AbstractBackend end

"""
    ExaTronBackend <: AbstractBackend

Solve OPF by decomposition using ExaTron.
"""
struct ExaTronBackend <: AbstractBackend end


"""
    AbstractBlockModel

Abstract supertype for the definition of block subproblems.
"""
abstract type AbstractBlockModel end

"""
    init!(block::AbstractBlockModel, algparams::AlgParams)

Init the optimization model by populating the model
with variables and constraints.

"""
function init! end

"""
    optimize!(block::AbstractBlockModel, x0::AbstractArray, algparams::AlgParams)

Solve the optimization problem, starting from an initial
variable `x0`. The optimization solver is specified in field
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
        primal::AbstractPrimalSolution,
        dual::AbstractDualSolution
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

struct EmptyBlockModel <: AbstractBlockModel end

function init!(block::EmptyBlockModel, algparams::AlgParams) end

function set_objective!(block::EmptyBlockModel, algparams::AlgParams,
                        primal::AbstractPrimalSolution, dual::AbstractDualSolution)
end
function get_solution(block::EmptyBlockModel) end

struct JuMPBlockBackend <: AbstractBlockModel
    id::Int
    k::Int
    t::Int
    model::JuMP.Model
    data::OPFData
    params::ModelInfo
    rawdata::RawData
end

### Implementation of JuMPBlockBackend
"""
    JuMPBlockBackend(
        blk::Int,
        opfdata::OPFData, raw_data::RawData,
        modelinfo::ModelInfo, t::Int, k::Int, T::Int,
    )

Use the modeler JuMP to define the optimal power flow
inside the block model.
This function is called inside the constructor
of the structure `OPFBlocks`, used for decomposition purpose.

# Arguments

- `blk::Int`: ID of the block represented by this model
- `opfdata::OPFData`: data used to build the optimal power flow problem.
- `raw_data::RawData`: same data, in raw format
- `modelinfo::ModelInfo`: parameters related to specification of the optimization model
- `t::Int`: current time-step. Value should be between `1` and `T`.
- `k::Int`: current contingency
- `T::Int`: final horizon

"""
function JuMPBlockBackend(
    blk::Int,
    opfdata::OPFData, raw_data::RawData, algparams::AlgParams,
    modelinfo::ModelInfo, t::Int, k::Int, T::Int;
)
    model = JuMP.Model()
    return JuMPBlockBackend(blk, k, t, model, opfdata, modelinfo, raw_data)
end

function init!(block::JuMPBlockBackend, algparams::AlgParams)
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

    # Fix penalty vars to 0
    fix.(opfmodel[:Zt], 0; force = true)
    if algparams.decompCtgs
        fix.(opfmodel[:Zk], 0; force = true)
    end

    # Add block constraints
    if !modelinfo.allow_constr_infeas
        zb = zeros(length(opfdata.buses))
        zl = zeros(length(opfdata.lines))
        σ_re = zb
        σ_im = zb
        σ_fr = zl
        σ_to = zl
    end
    @views for j=1:Kblock
        if modelinfo.allow_constr_infeas
            σ_re = opfmodel[:sigma_real][:,j,1]
            σ_im = opfmodel[:sigma_imag][:,j,1]
            σ_fr = opfmodel[:sigma_lineFr][:,j,1]
            σ_to = opfmodel[:sigma_lineTo][:,j,1]
        end
        opfdata_c = (j == 1) ? opfdata :
            opf_loaddata(block.rawdata;
            lineOff = opfdata.lines[block.rawdata.ctgs_arr[j - 1]],
            time_horizon_start = t,
            time_horizon_end = t,
            load_scale = modelinfo.load_scale,
            ramp_scale = modelinfo.ramp_scale,
            corr_scale = modelinfo.corr_scale)
        opf_model_add_real_power_balance_constraints(opfmodel, opfdata_c, opfmodel[:Pg][:,j,1], opfdata_c.Pd[:,1], opfmodel[:Vm][:,j,1], opfmodel[:Va][:,j,1], σ_re)
        opf_model_add_imag_power_balance_constraints(opfmodel, opfdata_c, opfmodel[:Qg][:,j,1], opfdata_c.Qd[:,1], opfmodel[:Vm][:,j,1], opfmodel[:Va][:,j,1], σ_im)
        if modelinfo.allow_line_limits
            opf_model_add_line_power_constraints(opfmodel, opfdata_c, opfmodel[:Vm][:,j,1], opfmodel[:Va][:,j,1], σ_fr, σ_to)
        end
    end
    return opfmodel
end

function set_objective!(block::JuMPBlockBackend, algparams::AlgParams,
                        primal::AbstractPrimalSolution, dual::AbstractDualSolution)
    blk = block.id
    opfmodel = block.model
    opfdata = block.data
    modelinfo = block.params
    k, t = block.k, block.t

    obj_expr = compute_objective_function(opfmodel, opfdata, modelinfo, algparams, k, t)
    auglag_penalty = opf_block_get_auglag_penalty_expr(
        opfmodel, modelinfo, opfdata, k, t, algparams, primal, dual)
    @objective(opfmodel, Min, obj_expr + auglag_penalty)
    return
end

function get_solution(block::JuMPBlockBackend)
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
        zt=JuMP.value.(opfmodel[:Zt]),
        sk=JuMP.value.(opfmodel[:Sk]),
        zk=JuMP.value.(opfmodel[:Zk]),
    )
    return solution
end

function set_start_values!(block::JuMPBlockBackend, x0)
    JuMP.set_start_value.(all_variables(block.model)[1:length(x0)], x0)
end

function optimize!(block::JuMPBlockBackend, x0::AbstractArray, algparams::AlgParams)
    blk = block.id
    opfmodel = block.model
    set_start_values!(block, x0)
    JuMP.optimize!(block.model)
    return get_solution(block)
end

function add_variables!(block::JuMPBlockBackend, algparams::AlgParams)
    opf_model_add_variables(
        block.model, block.data, block.params, algparams, block.k, block.t
    )
end

function add_ctgs_linking_constraints!(block::JuMPBlockBackend, algparams)
    opf_model_add_ctgs_linking_constraints(
        block.model, block.data, block.params,
    )
end

### Implementation of ExaBlockBackend
"""
    ExaBlockBackend(
        blk::Int,
        opfdata::OPFData, raw_data::RawData,
        modelinfo::ModelInfo, t::Int, k::Int, T::Int;
        device::TargetDevice=CPU,
        nr_tol::Float64=1e-10,
    )

Use the package ExaPF to define the optimal power flow
inside the block model.
This function is called inside the constructor
of the structure `OPFBlocks`, used for decomposition purpose.

# Arguments

- `blk::Int`: ID of the block represented by this model
- `opfdata::OPFData`: data used to build the optimal power flow problem.
- `raw_data::RawData`: same data, in raw format
- `modelinfo::ModelInfo`: parameters related to specification of the optimization model
- `t::Int`: current time-step. Value should be between `1` and `T`.
- `k::Int`: current contingency
- `T::Int`: final horizon

"""
struct ExaBlockBackend <: AbstractBlockModel
    id::Int
    k::Int
    t::Int
    model::ExaPF.AbstractNLPEvaluator
    data::OPFData
    params::ModelInfo
end

function ExaBlockBackend(
    blk::Int,
    opfdata::OPFData, raw_data::RawData, algparams::AlgParams,
    modelinfo::ModelInfo, t::Int, k::Int, T::Int,
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
    target = if algparams.device == CPU
        ExaPF.CPU()
    elseif algparams.device == CUDADevice
        ExaPF.CUDADevice()
    end
    model = ExaPF.ProxALEvaluator(power_network, time;
                                  device=target)
    return ExaBlockBackend(blk, k, t, model, opfdata, modelinfo)
end

function init!(block::ExaBlockBackend, algparams::AlgParams)
    opfmodel = block.model
    baseMVA = block.data.baseMVA

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

function add_ctgs_linking_constraints!(block::ExaBlockBackend)
    error("Contingencies are not supported in ExaPF")
end

function update_penalty!(block::ExaBlockBackend, algparams::AlgParams,
                         primal::AbstractPrimalSolution, dual::AbstractDualSolution)
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
        examodel.ρf = algparams.ρ_t
    end

    # Update next values
    if time != ExaPF.Final
        λt = dual.ramping[:, t+1]
        pgt = primal.Pg[:, 1, t+1] .- primal.St[:, t+1] .- primal.Zt[:, t+1] .+ ramp_agc
        ExaPF.update_multipliers!(examodel, ExaPF.Next(), λt)
        ExaPF.update_primal!(examodel, ExaPF.Next(), pgt)
        # Update parameters
        examodel.ρt = algparams.ρ_t
    end
end

function set_objective!(block::ExaBlockBackend, algparams::AlgParams,
                        primal::AbstractPrimalSolution, dual::AbstractDualSolution)
    update_penalty!(block, algparams, primal, dual)
    return
end

function get_solution(block::ExaBlockBackend, output)
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

function set_start_values!(block::ExaBlockBackend, x0)
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

function optimize!(block::ExaBlockBackend, x0::Union{Nothing, AbstractArray}, algparams::AlgParams)
    blk = block.id
    opfmodel = block.model
    optimizer = algparams.gpu_optimizer

    if isa(optimizer, MOI.OptimizerWithAttributes)
        optimizer = MOI.instantiate(optimizer)
    end

    # Critical part: if x0 is not feasible, it is very likely
    # that the Newton-Raphson algorithm implemented inside ExaPF
    # would fail to converge. If we do not trust x0, it is better
    # to pass nothing so ExaPF will compute a default starting point
    # on its own.
    if isa(x0, Array)
        set_start_values!(block, x0)
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


### Implementation of TronBlockBackend
"""
    TronBlockBackend(
        blk::Int,
        opfdata::OPFData, raw_data::RawData,
        modelinfo::ModelInfo, t::Int, k::Int, T::Int;
    )


# Arguments

- `blk::Int`: ID of the block represented by this model
- `opfdata::OPFData`: data used to build the optimal power flow problem.
- `raw_data::RawData`: same data, in raw format
- `modelinfo::ModelInfo`: parameters related to specification of the optimization model
- `t::Int`: current time-step. Value should be between `1` and `T`.
- `k::Int`: current contingency
- `T::Int`: final horizon

"""
struct TronBlockBackend <: AbstractBlockModel
    env::ExaTron.AdmmEnv
    id::Int
    k::Int
    t::Int
    T::Int
    data::OPFData
    params::ModelInfo
    tron_scale::Float64
    objective_scaling::Float64
end

# Map ProxAL's data to ExaTron's data
Base.convert(::Type{ExaTron.Bus}, b::Bus) =
    ExaTron.Bus(
        b.bus_i, b.bustype, b.Pd, b.Qd, b.Gs, b.Bs, b.area,
        b.Vm, b.Va, b.baseKV, b.zone, b.Vmax, b.Vmin,
    )

Base.convert(::Type{ExaTron.Line}, l::Line) =
    ExaTron.Line(
        l.from, l.to, l.r, l.x, l.b, l.rateA, l.rateB, l.rateC,
        l.ratio, l.angle, l.status, l.angmin, l.angmax,
    )

Base.convert(::Type{ExaTron.Gener}, g::Gener) =
    ExaTron.Gener(
        g.bus, g.Pg, g.Qg, g.Qmax, g.Qmin, g.Vg, g.mBase, g.status,
        g.Pmax, g.Pmin, g.Pc1, g.Pc2, g.Qc1min, g.Qc1max,
        g.Qc2min, g.Qc2max, g.ramp_agc, g.gentype, g.startup,
        g.shutdown, g.n, g.coeff,
    )

function ExaTron.OPFData(data::OPFData)
    fromlines, tolines = ExaTron.mapLinesToBuses(data.buses, data.lines, data.BusIdx)
    busGenerators = ExaTron.mapGenersToBuses(data.buses, data.generators, data.BusIdx)
    return ExaTron.OPFData(
        data.buses, data.lines, data.generators, data.bus_ref, data.baseMVA,
        data.BusIdx, fromlines, tolines, busGenerators,
    )
end

function TronBlockBackend(
    blk::Int,
    opfdata::OPFData, raw_data::RawData, algparams::AlgParams,
    modelinfo::ModelInfo, t::Int, k::Int, T::Int;
)
    scale = 1e-4
    use_gpu = (algparams.device == CUDADevice)
    trondata = ExaTron.OPFData(opfdata)
    env = ExaTron.ProxALAdmmEnv(
        trondata, use_gpu, t, T, algparams.tron_rho_pq, algparams.tron_rho_pa;
        allow_infeas=modelinfo.allow_constr_infeas, rho_sigma=modelinfo.weight_constr_infeas,
        verbose=algparams.verbose_inner, use_twolevel=true, outer_eps=algparams.tron_outer_eps,
    )
    return TronBlockBackend(env, blk, k, t, T, opfdata, modelinfo, scale, modelinfo.obj_scale)
end

function init!(block::TronBlockBackend, algparams::AlgParams)
    opfmodel = block.env
    baseMVA = block.data.baseMVA

    # Get params
    opfdata = block.data
    modelinfo = block.params
    Kblock = modelinfo.num_ctgs + 1
    t, k = block.t, block.k
    # Generators
    gens = block.data.generators

    # Sanity check
    @assert modelinfo.num_time_periods == 1
    @assert !algparams.decompCtgs || Kblock == 1

    # TODO: currently, only one contingency is supported
    j = 1
    pd = opfdata.Pd[:, 1]
    qd = opfdata.Qd[:, 1]
    # Pass load to exatron
    ExaTron.set_active_load!(opfmodel, pd)
    ExaTron.set_reactive_load!(opfmodel, qd)

    # Set bounds on slack variables s
    if algparams.decompCtgs && modelinfo.ctgs_link_constr_type == :corrective_penalty && k > 1
        ExaTron.set_upper_bound_slack!(opfmodel, 2 .* [g.scen_agc for g in gens])
    else
        ExaTron.set_upper_bound_slack!(opfmodel, 2 .* [g.ramp_agc for g in gens])
    end

    return opfmodel
end

function set_objective!(block::TronBlockBackend, algparams::AlgParams,
                        primal::AbstractPrimalSolution, dual::AbstractDualSolution)
    examodel = block.env
    opfdata = block.data
    modelinfo = block.params
    # Generators
    gens = block.data.generators
    ngen = length(gens)

    t, k = block.t, block.k
    ramp_agc = [g.ramp_agc for g in gens]

    # NOTE: ExaTron is solving the unscaled problem.
    # Need to prescale the penalty and the multipliers
    σ = block.objective_scaling

    #=
    # Old interface: this dead code block will be removed eventually
    # Update current values
    pgc = primal.Pg[:, k, t]
    ExaTron.set_proximal_ref!(examodel, pgc)
    ExaTron.set_proximal_term!(examodel, algparams.τ / σ)
    ExaTron.set_penalty!(examodel, algparams.ρ_t / σ)

    # Update previous values
    if t > 1
        λf = dual.ramping[:, t] ./ σ
        ExaTron.set_multiplier_last!(examodel, λf)
        pgf = primal.Pg[:, 1, t-1] .+ primal.Zt[:, t] .- ramp_agc
        ExaTron.set_proximal_last!(examodel, pgf)
    end

    # Update next values
    if t < block.T
        λt = dual.ramping[:, t+1] ./ σ
        pgt = primal.Pg[:, 1, t+1] .- primal.St[:, t+1] .- primal.Zt[:, t+1] .+ ramp_agc
        ExaTron.set_multiplier_next!(examodel, λt)
        ExaTron.set_proximal_next!(examodel, pgt)
    end
    =#

    # Set ExaTron's internal penalties to 0
    # (we are defining the penalties manually)
    examodel.model.gen_mod.rho = 0.0
    examodel.model.gen_mod.tau = 0.0
    # Reset Q_ref and c_ref
    examodel.model.gen_mod.Q_ref .= 0.0
    examodel.model.gen_mod.c_ref .= 0.0

    Q_ref = zeros(size(examodel.model.gen_mod.Q_ref))
    c_ref = zeros(size(examodel.model.gen_mod.c_ref))

    index_geners_Q = 1:4:4*ngen
    index_geners_c = 1:2:2*ngen
    index_slacks_c = 2:2:2*ngen

    # proximal terms
    pg_ref = primal.Pg[:, k, t]
    Q_ref[index_geners_Q] .+= algparams.τ / σ
    # Move to the GPU
    c_ref[index_geners_c] .-= algparams.τ .* pg_ref ./ σ

    # remove generation cost
    if (k == 1 && !modelinfo.allow_obj_gencost) ||
        (algparams.decompCtgs && modelinfo.ctgs_link_constr_type == :corrective_penalty && k > 1)
        alpha = [g.coeff[g.n-2]*opfdata.baseMVA^2 for g in gens]
        beta = [g.coeff[g.n-1]*opfdata.baseMVA for g in gens]
        Q_ref[index_geners_Q] .-= 2.0 * alpha
        c_ref[index_geners_c] .-= beta
    end

    # Ramping constraints (t-1, t)
    if t > 1 && k == 1
        λf = dual.ramping[:, t] ./ σ
        pgf = primal.Pg[:, 1, t-1] .+ primal.Zt[:, t] .- ramp_agc
        Q_ref .+= algparams.ρ_t*repeat([1., -1., -1., 1.], ngen)/σ
        c_ref[index_geners_c] .-= (pgf .* algparams.ρ_t ./ σ) .+ λf
        c_ref[index_slacks_c] .+= (pgf .* algparams.ρ_t ./ σ) .+ λf
    end

    # Ramping constraints (t, t+1)
    if t < block.T && k == 1
        λt = dual.ramping[:, t+1] ./ σ
        pgt = primal.Pg[:, 1, t+1] .- primal.St[:, t+1] .- primal.Zt[:, t+1] .+ ramp_agc
        Q_ref[index_geners_Q] .+= algparams.ρ_t / σ
        c_ref[index_geners_c] .+= -(pgt .*algparams.ρ_t ./ σ) .+ λt
    end

    # Contingency linking constraints
    if algparams.decompCtgs && modelinfo.ctgs_link_constr_type == :corrective_penalty
        scen_agc = [g.scen_agc for g in gens]
        K = size(primal.Pg, 2)

        # base case
        if k == 1 && K > 1
            λc = dropdims(sum(dual.ctgs[:, 2:K, t] ./ σ; dims = 2); dims=2)
            pgc = dropdims(sum(primal.Pg[:, 2:K, t] .- primal.Sk[:, 2:K, t] .- primal.Zk[:, 2:K, t] .+ scen_agc; dims = 2); dims=2)
            Q_ref[index_geners_Q] .+= (K - 1)*algparams.ρ_c / σ
            c_ref[index_geners_c] .+= -(pgc .* algparams.ρ_c ./ σ) .+ λc
        end

        # contingencies
        if k > 1
            λf = dual.ctgs[:, k, t] ./ σ
            pgf = primal.Pg[:, 1, t] .+ primal.Zk[:, k, t] .- scen_agc
            Q_ref .+= algparams.ρ_c .* repeat([1., -1., -1., 1.], ngen) ./ σ
            c_ref[index_geners_c] .-= (pgf .* algparams.ρ_c ./ σ) .+ λf
            c_ref[index_slacks_c] .+= (pgf .* algparams.ρ_c ./ σ) .+ λf
        end
    end
    # Copy to GPU
    copyto!(examodel.model.gen_mod.Q_ref, Q_ref)
    copyto!(examodel.model.gen_mod.c_ref, c_ref)
end

function get_solution(block::TronBlockBackend, output)

    # TODO: parse ExaTron solution
    exatron_status = output.status
    status = if exatron_status == ExaTron.HAS_CONVERGED
        MOI.OPTIMAL
    elseif exatron_status == ExaTron.MAXIMUM_ITERATIONS
        MOI.ITERATION_LIMIT
    end

    if status ∉ MOI_OPTIMAL_STATUSES
        @warn("Block $(block.id) subproblem not solved to optimality. status: $status")
    end

    s = ExaTron.slack_values(block.env) |> Array
    model = block.env.model
    solution = (
        status=status,
        minimum=block.objective_scaling * output.objval,
        pg=ExaTron.active_power_generation(model, output) |> Array,
        qg=ExaTron.reactive_power_generation(model, output) |> Array,
        vm=ExaTron.voltage_magnitude(model, output) |> Array,
        va=ExaTron.voltage_angle(model, output) |> Array,
        ωt=[0.0], # At the moment, no frequency variable in ExaTron
        st=(block.k > 1) ? (s .* 0) : s,
        sk=(block.k > 1) ? s : (s .* 0),
    )
    return solution
end

function set_start_values!(block::TronBlockBackend, x0)
    ngen = length(block.data.generators)
    nbus = length(block.data.buses)
    # Only one contingency, at the moment
    K = 1
    # Extract values from array x0
    pg = x0[1:ngen*K]
    qg = x0[ngen*K+1:2*ngen*K]
    vm = x0[2*ngen*K+1:2*ngen*K+nbus*K]
    va = x0[2*ngen*K+nbus*K+1:2*ngen*K+2*nbus*K]
    # Pass to ExaTron
    ExaTron.set_active_power_generation!(block.env, pg)
    ExaTron.set_reactive_power_generation!(block.env, qg)
    ExaTron.set_voltage_magnitude!(block.env, vm)
    ExaTron.set_voltage_angle!(block.env, va)
    return
end

function optimize!(block::TronBlockBackend, x0::Union{Nothing, AbstractArray}, algparams::AlgParams)
    if isa(x0, Array)
        set_start_values!(block, x0)
    end
    # Optimize with optimizer, using ExaPF model
    ExaTron.admm_restart!(block.env;
        scale=algparams.tron_scale, outer_iterlim=algparams.tron_outer_iterlim, inner_iterlim=algparams.tron_inner_iterlim
    )
    # Recover solution in ProxAL format
    solution = get_solution(block, block.env.solution)

    return solution
end
