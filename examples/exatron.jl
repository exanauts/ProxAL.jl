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
using Logging

MPI.Init()

# Select one of the following
# (case, T, ramp_scale, load_scale, maxρ, quad_penalty) = ("case_ACTIVSg2000_Corrected", 20, 0.3, 1.0, 0.1, 1e5) # 290 iterations
# (case, T, ramp_scale, load_scale, maxρ, quad_penalty) = ("case_ACTIVSg2000_Corrected", 20, 0.3, 1.0, 0.01, 1e3) # 449 iterations
# (case, T, ramp_scale, load_scale, maxρ, quad_penalty) = ("case_ACTIVSg2000_Corrected", 20, 0.3, 1.0, 0.01, 1e5) # 521 iterations
(case, T, ramp_scale, load_scale, maxρ, quad_penalty) = ("case_ACTIVSg2000", 2, 0.3, 1.0, 0.1, 1e3) # 225 iterations
# (case, T, ramp_scale, load_scale, maxρ, quad_penalty) = ("case9241pegase", 2, 0.3, 0.8, 0.01, 1e3)
# (case, T, ramp_scale, load_scale, maxρ, quad_penalty) = ("case118", 168, 0.2, 1.0, 0.1, 1e5)
# (case, T, ramp_scale, load_scale, maxρ, quad_penalty) = ("case9", 6, 0.04, 1.0, 0.1, 0.1)

T = parse(Int, ARGS[1])
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
  println("$T periods")
end

# No contingencies in this example
K = 0

# Load case
DATA_DIR = joinpath(dirname(@__FILE__), "..", "ExaData")
case_file = joinpath(DATA_DIR, "$(case).m")
load_file = joinpath(DATA_DIR, "$(case)")
# load_file = joinpath(DATA_DIR, "mp_demand", "$(case)_3600")

# Model/formulation settings
modelinfo = ModelInfo()
modelinfo.num_time_periods = T
modelinfo.load_scale = load_scale
modelinfo.ramp_scale = ramp_scale
modelinfo.allow_obj_gencost = true
modelinfo.allow_constr_infeas = false
modelinfo.weight_freq_ctrl = quad_penalty
modelinfo.time_link_constr_type = :penalty
modelinfo.ctgs_link_constr_type = :frequency_ctrl
modelinfo.case_name = case
modelinfo.num_ctgs = K

# Algorithm settings
algparams = AlgParams()
algparams.verbose = 1
algparams.tol = 1e-3
algparams.decompCtgs = true
algparams.iterlim = 100
algparams.device = ProxAL.ROCDevice
algparams.optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
algparams.tron_rho_pq=5*1e4
algparams.tron_rho_pa=5*1e5
algparams.tron_outer_iterlim=30
algparams.tron_inner_iterlim=2000
algparams.tron_scale=1e-5
algparams.mode = :coldstart
algparams.init_opf = false

# case_ACTIVSg2000_Corrected parameters
algparams.tron_rho_pq=5*1e4
algparams.tron_rho_pa=5*1e5
algparams.tron_outer_iterlim=30
algparams.tron_inner_iterlim=2000
algparams.tron_scale=1e-5

opt_sol, lyapunov_sol = Dict(), Dict()

ranks = MPI.Comm_size(MPI.COMM_WORLD)
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
   println("ProxAL/ExaTron $ranks ranks, $T periods")
end
cur_logger = global_logger(NullLogger())
elapsed_t = @elapsed begin
  redirect_stdout(devnull) do
    global nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, ProxAL.ExaTronBackend(), opt_sol, lyapunov_sol)
  end
end
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    global_logger(cur_logger)
    println("Creating problem: $elapsed_t")
    println("Benchmark Start")
    np = MPI.Comm_size(MPI.COMM_WORLD)
    elapsed_t = @elapsed begin
      info = ProxAL.optimize!(nlp)
    end
    println("AugLag iterations: $(info.iter) with $np ranks in $elapsed_t seconds")
else
  info = ProxAL.optimize!(nlp)
end

MPI.Finalize()
