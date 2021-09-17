using Test
using ProxAL
using DelimitedFiles, Printf
using LinearAlgebra, JuMP
using CatViews
using CUDA
using MPI

MPI.Init()

DATA_DIR = joinpath(dirname(@__FILE__), "..", "data")
case = "case9"
T = 2
ramp_scale = 0.5
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
modelinfo.weight_freq_ctrl = 0.1
modelinfo.time_link_constr_type = :penalty

# Algorithm settings
algparams = AlgParams()
algparams.verbose = 0

modelinfo.case_name = case
using Ipopt
algparams.optimizer =
	optimizer_with_attributes(Ipopt.Optimizer,
		"print_level" => Int64(algparams.verbose > 0)*5)
K = 1
modelinfo.num_ctgs = K
OPTIMAL_OBJVALUE = round(11258.316096601551*modelinfo.obj_scale, digits = 6)
OPTIMAL_PG = round.([0.8979870771416967, 1.3432060071608862, 0.9418738040358301, 0.9840203279072699, 1.448040097565594, 1.0149638851897345], digits = 5)
OPTIMAL_WT = round.([0.0, -0.0001206958510371199, 0.0, -0.0001468516901269684], sigdigits = 4)
proxal_ctgs_link = :frequency_penalty
algparams.decompCtgs = true
modelinfo.ctgs_link_constr_type = proxal_ctgs_link
algparams.mode = :coldstart
algparams.iterlim = 10
nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, JuMPBackend(), Dict(), Dict())
runinfo = ProxAL.optimize!(nlp)
@test isapprox(runinfo.objvalue[end], OPTIMAL_OBJVALUE, rtol = 1e-2)
@test isapprox(runinfo.x.Pg[:,1,:][:], OPTIMAL_PG, rtol = 1e-2)
@test runinfo.maxviol_c[end] <= algparams.tol
@test runinfo.maxviol_t[end] <= algparams.tol
@test runinfo.maxviol_c_actual[end] <= algparams.tol
@test runinfo.maxviol_t_actual[end] <= algparams.tol
@test runinfo.maxviol_d[end] <= algparams.tol
@test runinfo.iter <= algparams.iterlim
MPI.Finalize()
