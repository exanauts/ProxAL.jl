#
## Constant params
const MOI_OPTIMAL_STATUSES = [
    MOI.OPTIMAL,
    MOI.ALMOST_OPTIMAL,
    MOI.LOCALLY_SOLVED,
    MOI.ALMOST_LOCALLY_SOLVED,
]

@enum(TargetDevice,
    CPU,
    CUDADevice,
    Mixed,
)

#
#
# Algorithmic parameters
#
"""
    AlgParams

Specifies ProxAL's algorithmic parameters.

| Parameter | Description | Default value |
| :--- | :--- | :--- |
| `decompCtgs::Bool` | if true: decompose across contingencies (along with time) | false
| `jacobi::Bool` |     if true: do Jacobi updates, else do Gauss-Siedel updates | true
| `parallel::Bool` |   run NLP subproblems in parallel (needs MPI) | true
| `iterlim::Int` |     maximum number of ProxAL iterations | 100
| `nlpiterlim::Int` |  maximum number of NLP subproblem iterations | 100
| `tol::Float64` |     tolerance used for ProxAL termination | 1.0e-4
| `zero::Float64` |    tolerance below which is regarded as zero | 1.0e-8
| `ρ_t::Any` |         AL parameters for ramp constraints (can be different for different constraints) | 1.0
| `ρ_c::Any` |         AL parameters for ctgs constraints (can be different for different constraints) | 1.0
| `maxρ_t::Float64` |  Maximum value of `ρ_t` | 1.0
| `maxρ_c::Float64` |  Maximum value of `ρ_c` | 1.0
| `updateρ_t::Bool` |  if true: dynamically update `ρ_t` | false
| `updateρ_c::Bool` |  if true: dynamically update `ρ_c` | false
| `ρ_t_tol::Any` |     Tolerance for dynamic update of `ρ_t` | 1.0e-3
| `ρ_c_tol::Any` |     Tolerance for dynamic update of `ρ_c` | 1.0e-3
| `τ::Float64`       | Proximal weight parameter | 3.0
| `θ::Float64`       | Relaxation parameter for update of dual variables | 1.0
| `updateτ::Bool` |    if true: dynamically update `τ` | false
| `verbose::Int` |     level of output: 0 (none), 1 (stdout), 2 (+plots), 3 (+outfiles) | 0
| `mode::Symbol` |     computation mode `∈ [:nondecomposed, :coldstart, :lyapunov_bound]` | `:nondecomposed`
| `optimizer::Any` |   NLP solver | `nothing`
| `gpu_optimizer::Any` | GPU-compatible NLP solver | `nothing`
| `nr_tol::Float64`    | Tolerance of the Newton-Raphson algorithm (used only in `ReducedSpace()` model) | 1e-10
| `device::TargetDevice` | Target device to deport the resolution of the optimization problem | CPU
"""
mutable struct AlgParams
    decompCtgs::Bool# decompose contingencies (along with time)
    jacobi::Bool    # if true: do jacobi, else do gauss-siedel
    parallel::Bool  # run NLP subproblems in parallel
    iterlim::Int    # maximum number of ADMM iterations
    nlpiterlim::Int # maximum number of NLP subproblem iterations
    tol::Float64    # tolerance used for ADMM termination
    zero::Float64   # tolerance below which is regarded as zero
    ρ_t::Any        # AL parameters for ramp constraints (can be different for different constraints)
    ρ_c::Any        # AL parameters for ctgs constraints (can be different for different constraints)
    maxρ_t::Float64 # Maximum value of ρ for ramp constraints
    maxρ_c::Float64 # Maximum value of ρ for ctgs constraints
    updateρ_t::Bool # Dynamically update ρ for ramp constraints
    updateρ_c::Bool # Dynamically update ρ for ctgs constraints
    ρ_t_tol::Any    # Tolerance for dynamic update of ρ for ramp constraints
    ρ_c_tol::Any    # Tolerance for dynamic update of ρ for ramp constraints
    τ::Float64      # Proximal coefficient
    θ::Float64      # Relaxation parameter for update of dual variables
    updateτ::Bool   # Dynamically update τ
    verbose::Int    # level of output: 0 (none), 1 (stdout), 2 (+plots), 3 (+outfiles)
    mode::Symbol    # computation mode [:nondecomposed, :coldstart, :lyapunov_bound]
    optimizer::Any  # NLP solver for fullmodel and subproblems
    gpu_optimizer::Any  # GPU-compatible NLP solver for fullmodel and subproblems
    nr_tol::Float64 # Tolerance of the Newton-Raphson algorithm used in resolution of ExaBlockModel backend
    device::TargetDevice

    function AlgParams()
        new(
            false,  # decompCtgs
            true,   # jacobi
            true,   # parallel
            100,    # iterlim
            100,    # nlpiterlim
            1e-4,   # tol
            1e-8,   # zero
            1.0,    # ρ_t
            1.0,    # ρ_c
            1.0,    # maxρ_t
            1.0,    # maxρ_c
            false,  # updateρ_t
            false,  # updateρ_c
            1e-3,   # ρ_t_tol
            1e-3,   # ρ_c_tol
            3.0,    # τ
            1.0,    # θ
            false,  # updateτ
            0,      # verbose
            :nondecomposed, # mode
            nothing, # optimizer
            nothing, # GPU optimizer
            1e-10,
            CPU,
        )
    end
end

"""
    ModelParams

Specifies the ACOPF model structure.

| Parameter | Description | Default value |
| :--- | :--- | :--- |
| `num_time_periods::Int` | number of time periods | 1
| `num_ctgs::Int` | number of line contingencies | 0
| `load_scale::Float64` | load multiplication factor | 1.0
| `ramp_scale::Float64` | multiply this with ``p_{g}^{max}`` to get generator ramping ``r_g`` | 1.0
| `obj_scale::Float64` | objective multiplication factor | 1.0e-3
| `allow_obj_gencost::Bool` | model generator cost | true
| `allow_constr_infeas::Bool` | allow constraint infeasibility | false
| `weight_constr_infeas::Float64` | quadratic penalty weight for constraint infeasibilities | 1.0
| `weight_freq_ctrl::Float64` | quadratic penalty weight for frequency violations | 1.0
| `weight_ctgs::Float64` | linear weight of contingency objective function | 1.0
| `weight_quadratic_penalty_time::Float64` | see [Formulation](@ref) | 1.0
| `weight_quadratic_penalty_ctgs::Float64` | see [Formulation](@ref) | 1.0
| `case_name::String` | name of case file | ""
| `savefile::String` | name of save file | ""
| `time_link_constr_type::Symbol` | `∈ [:penalty, :equality, :inequality]` see [Formulation](@ref) | `:penalty`
| `ctgs_link_constr_type::Symbol` | `∈ [:frequency_ctrl, :preventive_penalty, :preventive_equality, :corrective_penalty, :corrective_equality, :corrective_penalty]`, see [Formulation](@ref) | `:preventive_equality`
"""
mutable struct ModelParams
    num_time_periods::Int
    num_ctgs::Int
    load_scale::Float64
    ramp_scale::Float64
    obj_scale::Float64
    allow_obj_gencost::Bool
    allow_constr_infeas::Bool
    weight_constr_infeas::Float64
    weight_freq_ctrl::Float64
    weight_ctgs::Float64
    weight_quadratic_penalty_time::Float64
    weight_quadratic_penalty_ctgs::Float64
    case_name::String
    savefile::String
    time_link_constr_type::Symbol
    ctgs_link_constr_type::Symbol
    # rho related
    maxρ_t::Float64
    maxρ_c::Float64
    # Initialize block OPFs with base OPF solution
    init_opf::Bool


    function ModelParams()
        new(
            1,     # num_time_periods
            0,     # num_ctgs
            1.0,   # load_scale
            1.0,   # ramp_scale
            1e-3,  # obj_scale
            true,  # allow_obj_gencost
            false, # allow_constr_infeas
            1.0,   # weight_constr_infeas
            1.0,   # weight_freq_ctrl
            1.0,   # weight_ctgs
            1.0,   # weight_quadratic_penalty_time
            1.0,   # weight_quadratic_penalty_ctgs
            "",    # case_name
            "",    # savefile
            :penalty,               # time_link_constr_type [:penalty,
                                    #                        :equality,
                                    #                        :inequality]
            :preventive_equality,    # ctgs_link_constr_type [:frequency_ctrl,
                                    #                        :preventive_penalty,
                                    #                        :preventive_equality,
                                    #                        :corrective_penalty,
                                    #                        :corrective_equality,
                                    #                        :corrective_inequality]
            0.1,
            0.1,
            false
        )
    end
end

"""
    set_penalty!(algparams::AlgParams;
             ngen::Int,
             maxρ_t::Float64,
             maxρ_c::Float64,
             modelinfo::ModelParams)

Initialize `algparams` for an ACOPF instance with `ngen` generators,
maximum augmented lagrangian parameter value of
`maxρ_t` (for ramping constraints),
`maxρ_c` (for contingency constraints),
and with model parameters specified in `modelinfo`.
"""
function set_penalty!(
    algparams::AlgParams,
    ngen::Int,
    maxρ_t::Float64,
    maxρ_c::Float64,
    modelinfo::ModelParams
)
    algparams.updateρ_t = (modelinfo.time_link_constr_type == :inequality)
    algparams.updateρ_c = (modelinfo.ctgs_link_constr_type == :corrective_inequality)
    algparams.ρ_t = maxρ_t*ones(ngen, modelinfo.num_time_periods)
    algparams.ρ_c = maxρ_c*ones(ngen, modelinfo.num_ctgs + 1, modelinfo.num_time_periods)
    if algparams.updateρ_t
        algparams.ρ_t .= 0
        algparams.ρ_t_tol = 1e-3*ones(size(algparams.ρ_t))
    end
    if algparams.updateρ_c
        algparams.ρ_c .= 0
        algparams.ρ_c_tol = 1e-3*ones(size(algparams.ρ_c))
    end
    algparams.maxρ_t = maxρ_t
    algparams.maxρ_c = maxρ_c
    algparams.τ = algparams.jacobi ? ((algparams.decompCtgs && modelinfo.num_ctgs > 0) ?
                            3max(maxρ_t, maxρ_c) : 3maxρ_t) : 0.0
    return nothing
end

