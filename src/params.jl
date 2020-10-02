#
# Algorithmic parameters
#
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

    function AlgParams()
        new(false,  # decompCtgs
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
            nothing # optimizer
        )
    end
end

#
# Model parameters and modelinfo
#
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

    function ModelParams()
        new(1,     # num_time_periods
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
            :preventive_equality    # ctgs_link_constr_type [:frequency_ctrl,
                                    #                        :preventive_penalty,
                                    #                        :preventive_equality,
                                    #                        :corrective_penalty,
                                    #                        :corrective_equality,
                                    #                        :corrective_inequality]
        )
    end
end


function set_rho!(algparams::AlgParams;
                  ngen::Int,
                  maxρ_t::Float64,
                  maxρ_c::Float64,
                  modelinfo::ModelParams)
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

