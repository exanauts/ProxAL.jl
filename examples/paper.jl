########################
# ProxAL runs for paper
########################

using ProxAL
using DelimitedFiles, Printf
using LinearAlgebra, JuMP, Ipopt
using CatViews
using MPI
using Logging

MPI.Init()
ranks = MPI.Comm_size(MPI.COMM_WORLD)

if length(ARGS) == 2
    case = ARGS[1]
    T = parse(Int, ARGS[2])
end

test_cases = ["case9", "case118", "case1354pegase", "case2383wp", "case9241pegase", "case_ACTIVSg2000"]
if length(ARGS) != 2 || case ∉ test_cases
    println("Usage: [mpiexec -n nprocs] julia --project examples/paper.jl case T")
    println("")
    println("       case must be one of $(test_cases)")
    println("")
    println("")
    exit()
end

# choose backend
backend = ProxAL.JuMPBackend()

# Load case
DATA_DIR = joinpath(dirname(@__FILE__), "..", "data")
case_file = joinpath(DATA_DIR, "$(case).m")
load_file = joinpath(DATA_DIR, "mp_demand", "$(case)_oneweek_168")
if case == "case_ACTIVSg2000"
    DATA_DIR = joinpath(dirname(@__FILE__), "..", "ExaData")
    case_file = joinpath(DATA_DIR, "$(case).m")
    load_file = joinpath(DATA_DIR, "$(case)")
end

# set parameters
if case == "case9"
    load_scale = 1.0
    ramp_scale = 0.05
    rho0 = nothing
elseif case == "case118"
    load_scale = 1.0
    ramp_scale = 0.2
    rho0 = nothing
elseif case == "case1354pegase"
    load_scale = 1.0
    ramp_scale = 0.3
    rho0 = 1e-2
elseif case == "case2383wp"
    load_scale = 1.0
    ramp_scale = 0.3
    rho0 = 1e-2
elseif case == "case9241pegase"
    load_scale = 0.8
    ramp_scale = 0.6
    rho0 = 1e-4
else
    load_scale = 1.0
    ramp_scale = 0.05
    rho0 = nothing
end

# Model/formulation settings
modelinfo = ModelInfo()
modelinfo.num_time_periods = T
modelinfo.num_ctgs = 0
modelinfo.load_scale = load_scale
modelinfo.ramp_scale = ramp_scale
modelinfo.allow_obj_gencost = true
modelinfo.allow_constr_infeas = false
modelinfo.time_link_constr_type = :penalty
modelinfo.allow_line_limits = false
modelinfo.case_name = case

# Algorithm settings
algparams = AlgParams()
algparams.optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0) #,  "tol" => 1e-1*algparams.tol)

cur_logger = global_logger(NullLogger())

# dry run to compile everything
algparams.verbose = 0
algparams.iterlim = 1
redirect_stdout(devnull) do
    global elapsed_t_dry_run_create_problem = @elapsed begin
        nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, backend, Dict(), Dict())
        nlp.problem.initial_solve = false
    end
    global elapsed_t_dry_run_optimize = @elapsed ProxAL.optimize!(nlp)
end


# the actual run
algparams.verbose = 1
algparams.iterlim = 100
elapsed_t = @elapsed begin
    redirect_stdout(devnull) do
        global nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, backend, Dict(), Dict())
        nlp.problem.initial_solve = false
    end
end

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    global_logger(cur_logger)
    println("ProxAL $case, $ranks ranks, $T periods")
    println("Dry run (create problem + optimize): $elapsed_t_dry_run_create_problem + $elapsed_t_dry_run_optimize")
    println("Creating problem: $elapsed_t")
    println("Benchmark Start")
    np = MPI.Comm_size(MPI.COMM_WORLD)
    elapsed_t = @elapsed begin
        info = ProxAL.optimize!(nlp; ρ_t_initial = rho0)
    end
    println("AugLag iterations: $(info.iter) with $np ranks in $elapsed_t seconds")
    @show(info.maxviol_t)
    @show(info.maxviol_t_actual)
    @show(info.maxviol_d)
    @show(info.objvalue/modelinfo.obj_scale)
    @show(info.lyapunov/modelinfo.obj_scale)
else
    info = ProxAL.optimize!(nlp; ρ_t_initial = rho0)
end

MPI.Finalize()
