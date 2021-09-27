
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
modelinfo.weight_constr_infeas = 1e8;
modelinfo.allow_line_limits = false
modelinfo.time_link_constr_type = :penalty
modelinfo.obj_scale = 1.0

# Algorithm settings
algparams = AlgParams()
algparams.verbose = 1
algparams.tol = 1e-3
#algparams.device = ProxAL.CUDADevice
algparams.optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
algparams.tron_rho_pq=5*1e4
algparams.tron_rho_pa=5*1e5
algparams.tron_outer_iterlim=30
algparams.tron_inner_iterlim=2000
algparams.tron_scale=1e-5
algparams.mode = :coldstart
algparams.init_opf = false
algparams.tron_outer_eps = 1e-6
algparams.tron_outer_iterlim=1000

algparams.iterlim = 10


# Contingencies
K = 0
modelinfo.num_ctgs = K
OBJVALUE = 0.0
PG = []

@testset "$T-period, $K-ctgs, time_link=$(modelinfo.time_link_constr_type)" begin

    @testset "ProxAL + OldExaTron" begin
        algparams.mode = :coldstart
        nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, ProxAL.ExaTronBackend(), Dict(), Dict(), nothing)
        runinfo = ProxAL.optimize!(nlp)
        global OBJVALUE = runinfo.objvalue[end]
        global PG = runinfo.x.Pg[:]
        global ST = runinfo.x.St[:]
    end

    @testset "ProxAL + NewExaTron" begin
        algparams.mode = :coldstart
        nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, ProxAL.NewExaTronBackend(), Dict(), Dict(), nothing)
        runinfo = ProxAL.optimize!(nlp)
        @test isapprox(runinfo.objvalue[end], OBJVALUE, rtol = 1e-4)
        @test isapprox(runinfo.x.Pg[:], PG, rtol = 1e-4)
        @test isapprox(runinfo.x.St[:], ST, rtol = 1e-4)
    end
end

