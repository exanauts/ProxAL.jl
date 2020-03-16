#
# Algorithmic parameters
#

mutable struct AlgParams
    aladin::Bool   # if true: do ALADIN, else do prox-ALM
    jacobi::Bool   # (only for prox-ALM) if true: do jacobi, else do gauss-siedel
    iterlim::Int   # maximum number of iterations
    tol::Float64   # tolerance used for termination
    zero::Float64  # tolerance below which is regarded as zero
    ρ::Float64     # Augmented Lagrangian coefficient
    τ::Float64     # Proximal coefficient
    μ::Float64     # (only for ALADIN) Augmented Lagrangian coefficient in QP
    θ::Float64     # Relaxation parameter for update of dual variables
    updateρ::Bool  # Dynamically update ρ
    updateτ::Bool  # (only for Proximal ALM) Dynamically update τ
    ipoptiterlim::Int
    maxjacobiroundsperiter::Int
end

function initializeParams(ρ = 10.0; aladin::Bool, jacobi::Bool)
    iterlim = 2000
    tol = 1e-2
    zero = 1e-4
    if aladin
        τ = 0.0
        μ = ρ
        ipoptiterlim = 10000
    else
        τ = 2ρ
        μ = 0.0
        ipoptiterlim = 10000
    end
    θ = 1.0
    updateρ = false
    updateτ = false
    maxjacobiroundsperiter = 1
    return AlgParams(aladin, jacobi, iterlim, tol, zero, ρ, τ, μ, θ, updateρ, updateτ, ipoptiterlim, maxjacobiroundsperiter)
end

