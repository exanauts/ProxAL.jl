using Test
using ProxAL
using DelimitedFiles, Printf
using LinearAlgebra, JuMP
using CatViews
using CUDA
using AMDGPU
using MPI
using LazyArtifacts

use_MPI = !isempty(ARGS) && (parse(Bool, ARGS[1]) == true)
use_MPI && MPI.Init()
const DATA_DIR = joinpath(artifact"ExaData", "ExaData")
# N-2 case9 files
case = "case9_2"
T = 2
ramp_scale = 0.5
load_scale = 1.0
rtol = 1e-4

# Load case
case_file = joinpath(DATA_DIR, "mp_demand", "$(case).m")
load_file = joinpath(DATA_DIR, "mp_demand", "$(case)")

# Model/formulation settings
modelinfo = ModelInfo()
modelinfo.num_time_periods = T
modelinfo.load_scale = load_scale
modelinfo.ramp_scale = ramp_scale
modelinfo.allow_obj_gencost = true
modelinfo.allow_constr_infeas = false
modelinfo.weight_freq_ctrl = 0.1
modelinfo.time_link_constr_type = :penalty
modelinfo.obj_scale = 1e-4

# Algorithm settings
algparams = AlgParams()
algparams.verbose = 0

solver_list = ["ExaAdmmCPU"]

@testset "Test ProxAL on $(case)" begin
    modelinfo.case_name = case

    using Ipopt
    backend = JuMPBackend()
    algparams.optimizer =
        optimizer_with_attributes(Ipopt.Optimizer,
            "print_level" => Int64(algparams.verbose > 0)*5)
    K = 1
    modelinfo.num_ctgs = K
    # ctgs_link = :preventive_penalty
    OPTIMAL_OBJVALUE = []
    OPTIMAL_PG = []
    OPTIMAL_WT = []

    algparams.iterlim = 10000
    algparams.decompCtgs = true
    modelinfo.ctgs_link_constr_type = :corrective_penalty
    algparams.mode = :coldstart
    algparams.tol = 1e-3
    algparams.verbose = 0
    nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, backend, use_MPI ? MPI.COMM_WORLD : nothing)
    runinfo = ProxAL.optimize!(nlp)
    OPTIMAL_OBJVALUE = runinfo.objvalue[end]
    OPTIMAL_PG = runinfo.x.Pg[:,1,:][:]
    OPTIMAL_WT = runinfo.x.ωt[:]
    @test runinfo.maxviol_c[end] <= algparams.tol
    @test runinfo.maxviol_t[end] <= algparams.tol
    @test runinfo.maxviol_c_actual[end] <= algparams.tol
    @test runinfo.maxviol_t_actual[end] <= algparams.tol
    @test runinfo.maxviol_d[end] <= algparams.tol
    @test runinfo.iter <= algparams.iterlim
    algparams.decompCtgs = false
    algparams.verbose = 0
    nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, backend, use_MPI ? MPI.COMM_WORLD : nothing)
    runinfo = ProxAL.optimize!(nlp)
    @test_broken isapprox(runinfo.objvalue[end], OPTIMAL_OBJVALUE, rtol = 1e-2)
    @test_broken isapprox(runinfo.x.Pg[:,1,:][:], OPTIMAL_PG, rtol = 1e-1)
    @test isapprox(runinfo.maxviol_c[end], 0.0)
    @test isapprox(runinfo.maxviol_c_actual[end], 0.0)
    @test runinfo.maxviol_c_actual[end] <= algparams.tol
    @test runinfo.maxviol_t_actual[end] <= algparams.tol
    @test runinfo.maxviol_d[end] <= algparams.tol

    backend = AdmmBackend()
    algparams.tron_outer_iterlim=20
    algparams.tron_inner_iterlim=800
    algparams.tron_outer_eps=1e-6
    algparams.decompCtgs = true
    algparams.mode = :coldstart
    algparams.iterlim = 10000
    algparams.verbose = 0
    algparams.init_opf = false
    algparams.θ_c = 1.0
    algparams.ρ_c = 1.0
    # Reduce tolerance for N-2
    algparams.tol = 1e-3
    nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, backend, use_MPI ? MPI.COMM_WORLD : nothing)
    runinfo = ProxAL.optimize!(nlp)
    @test isapprox(runinfo.objvalue[end], OPTIMAL_OBJVALUE, rtol = 1e-2)
    @test isapprox(runinfo.x.Pg[:,1,:][:], OPTIMAL_PG, rtol = 1e-1)
    @test runinfo.maxviol_c[end] <= algparams.tol
    @test runinfo.maxviol_t[end] <= algparams.tol
    @test runinfo.maxviol_c_actual[end] <= algparams.tol
    @test runinfo.maxviol_t_actual[end] <= algparams.tol
    @test runinfo.maxviol_d[end] <= algparams.tol
    @test runinfo.iter <= algparams.iterlim
end

use_MPI && MPI.Finalize()
