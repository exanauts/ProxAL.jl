mutable struct AlgNLPData
    nlpmodel::Vector{JuMP.Model}
    colCount::Int64
    colIndex::Dict{String, Int64}
    colLower::Array{Float64, 2}
    colUpper::Array{Float64, 2}
    colValue::Array{Float64, 2}
end

#
# Algorithmic parameters
#

mutable struct AlgParams
    aladin::Bool   # if true: do ALADIN, else do prox-ALM
    jacobi::Bool   # (only for prox-ALM) if true: do jacobi, else do gauss-siedel
    iterlim::Int   # maximum number of iterations
    tol::Float64   # tolerance used for termination
    zero::Float64  # tolerance below which is regarded as zero
    ρ::Any         # Augmented Lagrangian coefficient (can be different for different constraints)
    τ::Float64     # Proximal coefficient
    μ::Float64     # (only for ALADIN) Augmented Lagrangian coefficient in QP
    θ::Float64     # Relaxation parameter for update of dual variables
    maxρ::Float64  # Maximum value of ρ for any constraint
    updateρ::Bool  # Dynamically update ρ
    updateτ::Bool  # (only for Proximal ALM) Dynamically update τ
    nlpiterlim::Int# Maximum # of iterations in NLP step
end

function initializeParams(maxρ::Float64; aladin::Bool, jacobi::Bool, options::Option = Option())
    iterlim = 100
    tol = 1e-2
    zero = 1e-4
    ρ = maxρ
    if aladin
        τ = 0.0
        μ = maxρ
        nlpiterlim = 10000
        updateρ = false
    else
        τ = 0
        if jacobi && !options.two_block
            τ = 3maxρ
        end
        μ = 0.0
        nlpiterlim = 10000
        # update AL parameter only if
        # coupling constraints are inequalities
        updateρ = false
        if options.sc_constr
            updateρ = !(options.freq_ctrl || options.two_block)
        end
        if options.has_ramping
            updateρ = !options.quadratic_penalty
        end
    end
    θ = 1.0
    updateτ = false
    return AlgParams(aladin, jacobi, iterlim, tol, zero, ρ, τ, μ, θ, maxρ, updateρ, updateτ, nlpiterlim)
end

