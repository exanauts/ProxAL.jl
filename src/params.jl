#
# Algorithmic parameters
#
mutable struct AlgParams
    jacobi::Bool    # if true: do jacobi, else do gauss-siedel
    iterlim::Int    # maximum number of iterations
    tol::Float64    # tolerance used for termination
    zero::Float64   # tolerance below which is regarded as zero
    ρ::Any          # Augmented Lagrangian coefficient (can be different for different constraints)
    τ::Float64      # Proximal coefficient
    θ::Float64      # Relaxation parameter for update of dual variables
    maxρ::Float64   # Maximum value of ρ for any constraint
    updateρ::Bool   # Dynamically update ρ
    updateτ::Bool   # (only for Proximal ALM) Dynamically update τ
    nlpiterlim::Int # Maximum # of iterations in NLP step
    mode::String    # computation mode (nondecomposed, coldstart, lyapunov_bound)

    function AlgParams()
        new(true,   # jacobi
            100,    # iterlim
            1e-4,   # tol
            1e-8,   # zero
            1.0,    # ρ
            3.0,    # τ
            1.0,    # θ
            1.0,    # maxρ
            false,  # updateρ
            false,  # updateτ
            1000,   # nlpiterlim
            "nondecomposed" # mode
        )
    end
end

#
# Model parameters and options
#
mutable struct ModelParams
    num_time_periods::Int
    num_ctgs::Int
    obj_gencost::Bool
    allow_constr_infeas::Bool
    allow_load_shed::Bool
    add_quadratic_penalty::Bool
    weight_quadratic_penalty::Float64
    weight_scencost::Float64
    weight_loadshed::Float64
    weight_freqctrl::Float64
    savefile::String
    ctgs_link_constr_type::String

    function ModelParams()
        new(1,     # num_time_periods
            0,     # num_ctgs            
            true,  # obj_gencost
            false, # allow_constr_infeas
            false, # allow_load_shed
            false, # add_quadratic_penalty
            1.0,   # weight_quadratic_penalty
            1.0,   # weight_scencost
            1.0,   # weight_loadshed
            1.0,   # weight_freqctrl
            "",    # savefile
            "preventive" # ctgs_link_constr_type (preventive, corrective, frequency)
        )
    end
end

#
# Algorithm data
#
mutable struct AlgNLPData
    nlpmodel::Vector{JuMP.Model}
    colCount::Int64
    colIndex::Dict{String, Int64}
    colLower::Array{Float64, 2}
    colUpper::Array{Float64, 2}
    colValue::Array{Float64, 2}
end

