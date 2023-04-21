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
using LazyArtifacts

MPI.Init()

case = "case9"
demandfiles = "$(case)_oneweek_168"
# Large 2000 bus case
# case = "case_ACTIVSg2000"
# demandfiles = "$(case)"
# CUDA.device!(1)
# Load case
const DATA_DIR = joinpath(artifact"ExaData", "ExaData")
# const DATA_DIR = "cases"
case_file = joinpath(DATA_DIR, "$(case).m")
load_file = joinpath(DATA_DIR, "mp_demand", demandfiles)
# load_file = joinpath(DATA_DIR, demandfiles)

# choose one of the following (K*T subproblems in each case)
if length(ARGS) == 0
    (T, K) = (4, 2)
    # (T, K) = (4, 16)
elseif length(ARGS) == 4
    case = ARGS[1]
    demandfiles = ARGS[2]
    T = parse(Int, ARGS[3])
    K = parse(Int, ARGS[4])
else
    println("Usage: [mpiexec -n nprocs] julia --project examples/exatron.jl [case demandfiles T K]")
    println("")
    println("       (case,demandfiles,T,K) defaults to (case9,case9_oneweek_168,2,1)")
    exit()
end

# choose backend
backend = ProxAL.JuMPBackend()
# With ExaAdmmBackend(), CUDABackend will used
# backend = ProxAL.AdmmBackend()

# Model/formulation settings
modelinfo = ModelInfo()
modelinfo.num_time_periods = T
modelinfo.load_scale = 1.0
modelinfo.ramp_scale = 0.2
modelinfo.corr_scale = 0.5
modelinfo.allow_obj_gencost = true
modelinfo.allow_constr_infeas = true
modelinfo.weight_constr_infeas = 1e1
modelinfo.time_link_constr_type = :penalty
modelinfo.ctgs_link_constr_type = :corrective_penalty
modelinfo.allow_line_limits = true
modelinfo.case_name = case
modelinfo.num_ctgs = K

# Algorithm settings
algparams = AlgParams()
algparams.verbose = 1
algparams.tol = 1e-3
algparams.decompCtgs = (K > 0)
algparams.iterlim = 10000
if isa(backend, ProxAL.AdmmBackend)
    algparams.device = ProxAL.CUDABackend
end
algparams.optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0) #,  "tol" => 1e-1*algparams.tol)
algparams.tron_rho_pq=3e3
algparams.tron_rho_va=3e4
algparams.tron_outer_iterlim=8
algparams.tron_inner_iterlim=250
algparams.mode = :coldstart
algparams.init_opf = false
algparams.tron_outer_eps = Inf


ranks = MPI.Comm_size(MPI.COMM_WORLD)
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    println("ProxAL/ExaTron $ranks ranks, $T periods, $K contingencies")
end
# cur_logger = global_logger(NullLogger())
elapsed_t = @elapsed begin
    # redirect_stdout(devnull) do
        global nlp = ProxALEvaluator(
        case_file,
        load_file,
        modelinfo,
        algparams,
        backend,
    )
    # end
end
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    # global_logger(cur_logger)
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

ProxAL.write(info, nlp, "solution_$(case)_$(ranks).h5")

if !isinteractive()
    MPI.Finalize()
end
