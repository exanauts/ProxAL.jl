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

# case = "case9"
case = "case_ACTIVSg2000"

# choose one of the following (K*T subproblems in each case)
if length(ARGS) == 0
    # (K, T) = (1, 4)
    (K, T) = (19, 2)
    # (K, T) = (1, 10)
    # (K, T) = (10, 10)
    # (K, T) = (10, 100)
    # (K, T) = (100, 10)
    # (K, T) = (100, 100)
elseif length(ARGS) == 2
    T = parse(Int, ARGS[1])
    K = parse(Int, ARGS[2])
else
    println("Usage: [mpiexec -n nprocs] julia --project examples/exatron.jl [T K]")
    println("")
    println("       (K,T) defaults to (0,10)")
    exit()
end

# choose backend
# backend = ProxAL.JuMPBackend()
# # With ExaTronBackend(), CUDADevice will used
backend = ProxAL.ExaTronBackend()


# Load case
DATA_DIR = joinpath(dirname(@__FILE__), "..", "ExaData")
case_file = joinpath(DATA_DIR, "$(case).m")
load_file = joinpath(DATA_DIR, "$(case)")

# Model/formulation settings
modelinfo = ModelInfo()
modelinfo.num_time_periods = T
modelinfo.load_scale = 1.0
modelinfo.ramp_scale = 0.2
modelinfo.corr_scale = 0.5
modelinfo.allow_obj_gencost = true
modelinfo.allow_constr_infeas = true
modelinfo.weight_constr_infeas = 1e8
modelinfo.time_link_constr_type = :penalty
modelinfo.ctgs_link_constr_type = :corrective_penalty
modelinfo.allow_line_limits = false
modelinfo.case_name = case
modelinfo.num_ctgs = K

# Algorithm settings
algparams = AlgParams()
algparams.verbose = 1
algparams.verbose_inner = 0
algparams.tol = 1e-3
algparams.decompCtgs = (K > 0)
algparams.iterlim = 100
if isa(backend, ProxAL.ExaTronBackend)
  algparams.device = ProxAL.ROCDevice
end
algparams.optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
algparams.tron_rho_pq=5*1e4
algparams.tron_rho_pa=5*1e5
algparams.tron_outer_iterlim=30
algparams.tron_inner_iterlim=2000
algparams.tron_scale=1e-5
algparams.mode = :coldstart
algparams.init_opf = false
algparams.tron_outer_eps = 1e-3


ranks = MPI.Comm_size(MPI.COMM_WORLD)
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
   println("ProxAL/ExaTron $ranks ranks, $T periods, $K contingencies")
end
cur_logger = global_logger(NullLogger())
elapsed_t = @elapsed begin
  redirect_stdout(devnull) do
    global nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, backend, Dict(), Dict())
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
