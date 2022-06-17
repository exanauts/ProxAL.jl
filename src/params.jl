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
| `num_sweeps::Int` |  number of jacobi/gauss-seidel sweeps per primal update step | 1
| `iterlim::Int` |     maximum number of ProxAL iterations | 100
| `nlpiterlim::Int` |  maximum number of NLP subproblem iterations | 100
| `tol::Float64` |     tolerance used for ProxAL termination | 1.0e-4
| `zero::Float64` |    tolerance below which is regarded as zero | 1.0e-8
| `θ_t::Float64` |     see [Formulation](@ref) | 1.0
| `θ_c::Float64` |     see [Formulation](@ref) | 1.0
| `ρ_t::Float64` |     AL penalty weight for ramp constraints | 1.0
| `ρ_c::Float64` |     AL penalty weight for ctgs constraints | 1.0
| `updateρ_t::Bool` |  if true: dynamically update `ρ_t` | true
| `updateρ_c::Bool` |  if true: dynamically update `ρ_c` | true
| `τ::Float64`       | Proximal weight parameter | 3.0
| `updateτ::Bool` |    if true: dynamically update `τ` | true
| `verbose::Int` |     level of output: 0 (none), 1 (stdout) | 0
| `mode::Symbol` |     computation mode `∈ [:nondecomposed, :coldstart, :lyapunov_bound]` | `:coldstart`
| `optimizer::Any` |   NLP solver | `nothing`
| `gpu_optimizer::Any` | GPU-compatible NLP solver | `nothing`
| `nr_tol::Float64`    | Tolerance of the Newton-Raphson algorithm (used only in `ExaPFBackend()` model) | 1e-10
| `init_opt::Bool` |   if true: initialize block OPFs with base OPF solution | false
| `device::TargetDevice` | Target device to deport the resolution of the optimization problem | CPU
| `verbose_inner::Int` | Verbose level for `ExaTronBackend()` | 0
| `tron_rho_pq::Float64` | Parameter for `ExaTronBackend()` | 4e2
| `tron_rho_pa::Float64` | Parameter for `ExaTronBackend()` | 4e4
| `tron_inner_iterlim::Int` | Parameter for `ExaTronBackend()` | 800
| `tron_outer_iterlim::Int` | Parameter for `ExaTronBackend()` | 20
| `tron_outer_eps::Float64` | Parameter for `ExaTronBackend()` | 1e-4
"""
Base.@kwdef mutable struct AlgParams
    decompCtgs::Bool = false # decompose contingencies (along with time)
    jacobi::Bool     = true  # if true: do jacobi, else do gauss-siedel
    num_sweeps::Int  = 1     # Number of jacobi/gauss-seidel sweeps per primal update step
    iterlim::Int     = 100   # maximum number of ADMM iterations
    nlpiterlim::Int  = 100   # maximum number of NLP subproblem iterations
    tol::Float64     = 1e-3  # tolerance used for ADMM termination
    zero::Float64    = 1e-8  # tolerance below which is regarded as zero
    θ_t::Float64     = 1.0   # weight_quadratic_penalty_time
    θ_c::Float64     = 1.0   # weight_quadratic_penalty_ctgs
    ρ_t::Float64     = 1.0   # AL parameter for ramp constraints
    ρ_c::Float64     = 1.0   # AL parameter for ctgs constraints
    updateρ_t::Bool  = true  # Dynamically update ρ for ramp constraints
    updateρ_c::Bool  = true  # Dynamically update ρ for ctgs constraints
    τ::Float64       = 3.0   # Proximal coefficient
    updateτ::Bool    = true  # Dynamically update τ
    verbose::Int     = 0     # level of output: 0 (none), 1 (stdout), 2 (+plots), 3 (+outfiles)
    mode::Symbol            = :coldstart     # computation mode [:nondecomposed, :coldstart, :lyapunov_bound]
    optimizer::Any          = nothing        # NLP solver for fullmodel and subproblems
    gpu_optimizer::Any      = nothing        # GPU-compatible NLP solver for fullmodel and subproblems
    nr_tol::Float64         = 1e-10          # Tolerance of the Newton-Raphson algorithm (for ExaBlockBackend backend)
    init_opf::Bool          = false
    device::TargetDevice    = CPU
    verbose_inner::Int      = 0
    tron_rho_pq::Float64    = 4e2
    tron_rho_pa::Float64    = 4e4
    tron_inner_iterlim::Int = 800
    tron_outer_iterlim::Int = 20
    tron_outer_eps::Float64 = 1e-4
end

"""
    ModelInfo

Specifies the ACOPF model structure.

| Parameter | Description | Default value |
| :--- | :--- | :--- |
| `time_horizon_start::Int` | starting index of planning horizon | 1
| `num_time_periods::Int` | number of time periods | 1
| `num_ctgs::Int` | number of line contingencies | 0
| `load_scale::Float64` | load multiplication factor | 1.0
| `ramp_scale::Float64` | multiply this with ``p_{g}^{max}`` to get generator ramping ``r_g`` | 1.0
| `corr_scale::Float64` | multiply this with ``r_g`` to get generator ramping for corrective control | 0.1
| `obj_scale::Float64` | objective multiplication factor | 1.0e-3
| `allow_obj_gencost::Bool` | model generator cost | true
| `allow_constr_infeas::Bool` | allow constraint infeasibility | false
| `allow_line_limits::Bool` | allow line flow limits | true
| `weight_constr_infeas::Float64` | quadratic penalty weight for constraint infeasibilities | 1.0
| `weight_freq_ctrl::Float64` | quadratic penalty weight for frequency violations | 1.0
| `weight_ctgs::Float64` | linear weight of contingency objective function | 1.0
| `case_name::String` | name of case file | ""
| `savefile::String` | name of save file | ""
| `time_link_constr_type::Symbol` | `∈ [:penalty, :equality, :inequality]` see [Formulation](@ref) | `:penalty`
| `ctgs_link_constr_type::Symbol` | `∈ [:frequency_penalty, :frequency_equality, :preventive_penalty, :preventive_equality, :corrective_penalty, :corrective_equality, :corrective_inequality]`, see [Formulation](@ref) | `:frequency_penalty`
"""
Base.@kwdef mutable struct ModelInfo
    time_horizon_start::Int = 1
    num_time_periods::Int = 1
    num_ctgs::Int = 0
    load_scale::Float64 = 1.0
    ramp_scale::Float64 = 1.0
    corr_scale::Float64 = 0.1
    obj_scale::Float64 = 1e-3
    allow_obj_gencost::Bool = true
    allow_constr_infeas::Bool = false
    allow_line_limits::Bool = true
    weight_constr_infeas::Float64 = 1.0
    weight_freq_ctrl::Float64 = 1.0
    weight_ctgs::Float64 = 1.0
    case_name::String = ""
    savefile::String = ""
    time_link_constr_type::Symbol = :penalty # ∈ [:penalty, :equality, :inequality]
    ctgs_link_constr_type::Symbol = :frequency_penalty # ∈ [:frequency_penalty, :frequency_equality, :preventive_penalty, :preventive_equality, :corrective_penalty, :corrective_equality, :corrective_inequality]
end
