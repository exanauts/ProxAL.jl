# using CUDA
using ProxAL
using LinearAlgebra
using Test
using LazyArtifacts
using ExaAdmm
using CUDA

CUDA.allowscalar(false)

import ProxAL: ExaAdmmBackend
const AB = ExaAdmmBackend

const DATA_DIR = joinpath(artifact"ExaData", "ExaData")
CASE = joinpath(DATA_DIR, "case9.m")

RAMP_AGC = [1.25, 1.5, 1.35]

LOADS = Dict(
    1 => Dict(
              "pd"=>[0.0, 0.0, 0.0, 0.0, 90.0, 0.0, 100.0, 0.0, 125.0],
              "qd"=>[0.0, 0.0, 0.0, 0.0, 30.0, 0.0, 35.0, 0.0, 50.0],
    ),
    2 => Dict(
              "pd"=>[0.0, 0.0, 0.0, 0.0, 97.3798938, 0.0, 108.199882, 0.0, 135.2498525],
              "qd"=>[0.0, 0.0, 0.0, 0.0, 32.4599646, 0.0, 37.8699587, 0.0, 54.099941],
    )
)

USE_GPUS = [false]
has_cuda_gpu() && push!(USE_GPUS, true)

@testset "ProxAL wrapper (CUDA=$use_gpu)" for use_gpu in USE_GPUS
    t, horizon = 1, 2
    rho_pq, rho_va = 400.0, 40000.0

    if use_gpu
        T = Float64
        VT = CuVector{Float64}
        VI = CuVector{Int}
        MT = CuMatrix{Float64}
    else
        T = Float64
        VT = Vector{Float64}
        VI = Vector{Int}
        MT = Matrix{Float64}
    end

    env = ExaAdmm.AdmmEnv{T,VT,VI,MT}(CASE, rho_pq, rho_va)
    model = AB.ModelProxAL(env, t, horizon)

    n = model.n
    env.params.inner_iterlim = 1000
    env.params.verbose = 0

    rho = rand()
    tau = rand()
    AB.set_proximal_term!(model, tau)
    @test model.tau == tau
    AB.set_penalty!(model, rho)
    @test model.rho == rho

    smax = RAMP_AGC
    AB.set_upper_bound_slack!(model, smax)
    AB.set_active_load!(model, LOADS[t]["pd"])
    AB.set_reactive_load!(model, LOADS[t]["qd"])

    # Test ExaAdmm interface
    # / preprocess
    ExaAdmm.admm_increment_outer(env, model)
    ExaAdmm.admm_outer_prestep(env, model)
    ExaAdmm.admm_increment_reset_inner(env, model)
    ExaAdmm.admm_increment_inner(env, model)
    ExaAdmm.admm_inner_prestep(env, model)
    # / update
    ExaAdmm.admm_update_x(env, model)
    ExaAdmm.admm_update_xbar(env, model)
    ExaAdmm.admm_update_z(env, model)
    ExaAdmm.admm_update_l(env, model)
    ExaAdmm.admm_update_residual(env, model)
    ExaAdmm.admm_update_lz(env, model)

    # Solve!
    ExaAdmm.admm_two_level(env, model)

    pg = AB.active_power_generation(model)
    qg = AB.reactive_power_generation(model)

    @test Array(pg) ≈ [0.898281257259199, 1.3402461278961415, 0.940057474370767] rtol=1e-5
    @test Array(qg) ≈ [0.13441293341552688, 0.07454842846214903, -0.15922687685240058] rtol=1e-5

    # TODO
    @test model.info.status == :Solved
end

