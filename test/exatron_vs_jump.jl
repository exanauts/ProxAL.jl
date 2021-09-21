
using Test
using ProxAL
using DelimitedFiles, Printf
using LinearAlgebra, JuMP
using Ipopt
using CatViews
using CUDA

DATA_DIR = joinpath(dirname(@__FILE__), "..", "data")
case = "case9"
T = 2
ramp_scale = 0.03
load_scale = 1.0
rtol = 1e-4

# Load case
case_file = joinpath(DATA_DIR, "$(case).m")
load_file = joinpath(DATA_DIR, "mp_demand", "$(case)_oneweek_168")
# ctgs_arr = deepcopy(rawdata.ctgs_arr)

# Model/formulation settings
modelinfo = ModelInfo()
modelinfo.num_time_periods = T
modelinfo.load_scale = load_scale
modelinfo.ramp_scale = ramp_scale
modelinfo.allow_obj_gencost = true
modelinfo.allow_constr_infeas = false
modelinfo.allow_line_limits = false
modelinfo.time_link_constr_type = :penalty

# Algorithm settings
algparams = AlgParams()
algparams.verbose = 1
algparams.tol = 1e-3
algparams.iterlim = 100
#algparams.device = ProxAL.CUDADevice
algparams.optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
algparams.tron_rho_pq=5*1e4
algparams.tron_rho_pa=5*1e5
algparams.tron_outer_iterlim=30
algparams.tron_inner_iterlim=2000
algparams.tron_scale=1e-5
algparams.mode = :coldstart
algparams.init_opf = false


# Contingencies
K = 0
modelinfo.num_ctgs = K
OPTIMAL_OBJVALUE = 0.0
OPTIMAL_PG = []

@testset "$T-period, $K-ctgs, time_link=$(modelinfo.time_link_constr_type)" begin
    
    @testset "Non-decomposed formulation" begin
        # modelinfo.ctgs_link_constr_type = :corrective_equality
        algparams.mode = :nondecomposed
        algparams.θ_t = algparams.θ_c = (10/algparams.tol)^2
        nlp = NonDecomposedModel(case_file, load_file, modelinfo, algparams)
        result = ProxAL.optimize!(nlp)
        global OPTIMAL_OBJVALUE = result["objective_value_nondecomposed"]
        global OPTIMAL_PG = result["primal"].Pg[:,1,:][:]
        @test norm(result["primal"].Zt[:], Inf) <= algparams.tol
        @test norm(result["primal"].Zk[:], Inf) <= algparams.tol
    end

    @testset "ProxAL + JuMP" begin
        # modelinfo.ctgs_link_constr_type = :corrective_penalty
        algparams.mode = :coldstart
        nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, JuMPBackend(), Dict(), Dict(), nothing)
        runinfo = ProxAL.optimize!(nlp)
        @test isapprox(runinfo.objvalue[end], OPTIMAL_OBJVALUE, rtol = 1e-2)
        @test isapprox(runinfo.x.Pg[:,1,:][:], OPTIMAL_PG, rtol = 1e-2)
        @test runinfo.maxviol_c[end] <= algparams.tol
        @test runinfo.maxviol_t[end] <= algparams.tol
        @test runinfo.maxviol_c_actual[end] <= algparams.tol
        @test runinfo.maxviol_t_actual[end] <= algparams.tol
        @test runinfo.maxviol_d[end] <= algparams.tol
        @test runinfo.iter <= algparams.iterlim
    end

    @testset "ProxAL + ExaTron" begin
        # modelinfo.ctgs_link_constr_type = :corrective_penalty
        algparams.mode = :coldstart
        # algparams.iterlim = 1000
        nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, ProxAL.ExaTronBackend(), Dict(), Dict(), nothing)
        runinfo = ProxAL.optimize!(nlp)
        @test isapprox(runinfo.objvalue[end], OPTIMAL_OBJVALUE, rtol = 1e-2)
        @test isapprox(runinfo.x.Pg[:,1,:][:], OPTIMAL_PG, rtol = 1e-2)
        @test runinfo.maxviol_c[end] <= algparams.tol
        @test runinfo.maxviol_t[end] <= algparams.tol
        @test runinfo.maxviol_c_actual[end] <= algparams.tol
        @test runinfo.maxviol_t_actual[end] <= algparams.tol
        @test runinfo.maxviol_d[end] <= algparams.tol
        @test runinfo.iter <= algparams.iterlim
    end
end

