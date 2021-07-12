using Test
using ProxAL
using DelimitedFiles, Printf
using LinearAlgebra, JuMP, Ipopt
using CatViews
using CUDA
using MPI

MPI.Init()
DATA_DIR = joinpath(dirname(@__FILE__), "..", "data")
case = "case9"
T = 2
K = 0
ramp_scale = 0.5
load_scale = 1.0
maxρ = 0.1
quad_penalty = 0.1
rtol = 1e-2

# Load case
case_file = joinpath(DATA_DIR, "$(case).m")
load_file = joinpath(DATA_DIR, "mp_demand", "$(case)_oneweek_168")
# ctgs_arr = deepcopy(rawdata.ctgs_arr)

# Model/formulation settings
modelinfo = ModelParams()
modelinfo.num_time_periods = T
modelinfo.load_scale = load_scale
modelinfo.ramp_scale = ramp_scale
modelinfo.allow_obj_gencost = true
modelinfo.allow_constr_infeas = false
modelinfo.time_link_constr_type = :penalty
modelinfo.ctgs_link_constr_type = :frequency_ctrl
modelinfo.case_name = case
modelinfo.num_ctgs = K

# Algorithm settings
algparams = AlgParams()
algparams.verbose = 0
algparams.init_opf = true
algparams.θ_t = quad_penalty
algparams.θ_c = quad_penalty
algparams.ρ_t = algparams.ρ_c = maxρ
algparams.τ = 3maxρ
algparams.decompCtgs = false

# For JuMP
algparams.optimizer = optimizer_with_attributes(
    Ipopt.Optimizer,
    "print_level" => Int64(algparams.verbose > 0)*5,
)
# For ExaPF
algparams.gpu_optimizer = optimizer_with_attributes(
    Ipopt.Optimizer,
    "print_level" => 0,
    "limited_memory_max_history" => 50,
    "hessian_approximation" => "limited-memory",
    "tol" => 1e-5,
)

@testset "Test ProxAL on $(case) with $T-period, $K-ctgs, time_link=penalty and Ipopt" begin
    algparams.mode = :coldstart
    nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, ProxAL.JuMPBackend())
    info = ProxAL.optimize!(nlp)
    # @test isapprox(info.maxviol_c[end], 0.0)
    @test isapprox(info.x.Pg[:], [0.8979849196165037, 1.3432106614001416, 0.9418713794662078, 0.9840203268799962, 1.4480400989162827, 1.0149638876932787], rtol = rtol)
    @test isapprox(info.λ.ramping[:], [0.0, 0.0, 0.0, 2.1600093405682597e-6, -7.2856620728201185e-6, 5.051385899057505e-6], rtol = rtol)
    @test isapprox(info.maxviol_t[end], 2.687848059435005e-5, rtol = rtol)
    @test isapprox(info.maxviol_d[end], 7.28542741650351e-6, rtol = rtol)
    @test info.iter == 5
end
MPI.Finalize()
