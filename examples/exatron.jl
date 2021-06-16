#############################################################################
# ProxAL/ExaTron Example File
# This example runs ProxAL with ExaTron as a backend and outputs a profile
# file. PProf has to be installed in the global environment.
#############################################################################

using ProxAL
using DelimitedFiles, Printf
using LinearAlgebra, JuMP, Ipopt
using CatViews
using CUDA
using MPI
using Profile
using PProf

MPI.Init()

# Select one of the following
(case, T, ramp_scale, load_scale, maxρ, quad_penalty) = ("case_ACTIVSg2000_Corrected", 3600, 0.3, 0.8, 0.01, 1e3)
(case, T, ramp_scale, load_scale, maxρ, quad_penalty) = ("case9241pegase", 20, 0.3, 0.8, 0.01, 1e3)
# (case, T, ramp_scale, load_scale, maxρ, quad_penalty) = ("case118", 168, 0.2, 1.0, 0.1, 1e5)
# (case, T, ramp_scale, load_scale, maxρ, quad_penalty) = ("case9", 6, 0.04, 1.0, 0.1, 0.1)

# No contingencies in this example
K = 0

# Load case
DATA_DIR = joinpath(dirname(@__FILE__), "..", "data")
case_file = joinpath(DATA_DIR, "$(case).m")
load_file = joinpath(DATA_DIR, "mp_demand", "$(case)_oneweek_168")

# Model/formulation settings
modelinfo = ModelParams()
modelinfo.num_time_periods = T
modelinfo.load_scale = load_scale
modelinfo.ramp_scale = ramp_scale
modelinfo.allow_obj_gencost = true
modelinfo.allow_constr_infeas = false
modelinfo.weight_quadratic_penalty_time = quad_penalty
modelinfo.weight_freq_ctrl = quad_penalty
modelinfo.time_link_constr_type = :penalty
modelinfo.ctgs_link_constr_type = :frequency_ctrl
modelinfo.case_name = case
modelinfo.num_ctgs = K
# rho related
modelinfo.maxρ_t = maxρ
modelinfo.maxρ_c = maxρ
# Initialize block OPFs with base OPF solution
modelinfo.init_opf = false

# Algorithm settings
algparams = AlgParams()
algparams.parallel = true #algparams.parallel = (nprocs() > 1)
algparams.verbose = 0
algparams.decompCtgs = false
algparams.iterlim = 1
algparams.device = ProxAL.CUDADevice
algparams.optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)

opt_sol, lyapunov_sol = Dict(), Dict()

ranks = MPI.Comm_size(MPI.COMM_WORLD)
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
   println("ProxAL/ExaTron $ranks ranks, $T periods")
end

algparams.mode = :coldstart
nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, ProxAL.ExaTronBackend(), opt_sol, lyapunov_sol)
info = ProxAL.optimize!(nlp)
nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, ProxAL.ExaTronBackend(), opt_sol, lyapunov_sol)
info = ProxAL.optimize!(nlp)
nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, ProxAL.ExaTronBackend(), opt_sol, lyapunov_sol)
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    np = MPI.Comm_size(MPI.COMM_WORLD)
    Profile.init(n = 10^7, delay = 0.001)
    @profile (global info = ProxAL.optimize!(nlp))
    println("AugLag iterations: $(info.iter) with $np ranks")
    pprof(web=false, out="prof_$ranks")
else
  info = ProxAL.optimize!(nlp)
end


MPI.Finalize()

