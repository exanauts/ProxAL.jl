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

function initializeParams(maxρ = 10.0; aladin::Bool, jacobi::Bool)
    iterlim = 2000
    tol = 1e-2
    zero = 1e-4
    ρ = maxρ
    if aladin
        τ = 0.0
        μ = maxρ
        nlpiterlim = 10000
        updateρ = false
    else
        τ = 10maxρ
        μ = 0.0
        nlpiterlim = 10000
        updateρ = true
    end
    θ = 1.0
    updateτ = false
    return AlgParams(aladin, jacobi, iterlim, tol, zero, ρ, τ, μ, θ, maxρ, updateρ, updateτ, nlpiterlim)
end

