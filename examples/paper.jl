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

if length(ARGS) == 4
    case = ARGS[1]
    ramp_scale = parse(Float64, ARGS[2])
    ρ_t_initial = parse(Float64, ARGS[3])
    τ_factor = parse(Float64, ARGS[4])
end

if length(ARGS) != 4 || case ∉ ["case118", "case1354pegase", "case9241pegase"]
    println("Usage: [mpiexec -n nprocs] julia --project examples/paper.jl case ramp_scale rho_initial tau_factor")
    println("")
    exit()
end

# choose backend
backend = ProxAL.JuMPBackend()

# Load case
DATA_DIR = joinpath(dirname(@__FILE__), "..", "data")
case_file = joinpath(DATA_DIR, "$(case).m")
load_file = joinpath(DATA_DIR, "mp_demand", "$(case)_oneweek_168")

# Model/formulation settings
modelinfo = ModelInfo()
modelinfo.num_time_periods = 168
modelinfo.num_ctgs = 0
modelinfo.load_scale = (case == "case9241pegase") ? 0.8 : 1.0
modelinfo.ramp_scale = ramp_scale
modelinfo.allow_obj_gencost = true
modelinfo.allow_constr_infeas = false
modelinfo.time_link_constr_type = :penalty
modelinfo.allow_line_limits = false
modelinfo.case_name = case
modelinfo.obj_scale = 1e-3

# Algorithm settings
algparams = AlgParams()
algparams.optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)

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
algparams.tol = 1e-3
elapsed_t = @elapsed begin
    redirect_stdout(devnull) do
        global nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, backend, Dict(), Dict())
        nlp.problem.initial_solve = false
    end
end

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    global_logger(cur_logger)
    println("ProxAL $case, $ranks ranks, $(modelinfo.num_time_periods) periods")
    println("Dry run (create problem + optimize): $elapsed_t_dry_run_create_problem + $elapsed_t_dry_run_optimize")
    println("Creating problem: $elapsed_t")
    println("Benchmark Start")
    np = MPI.Comm_size(MPI.COMM_WORLD)
    elapsed_t = @elapsed begin
        info = ProxAL.optimize!(nlp; ρ_t_initial = ρ_t_initial, τ_factor = τ_factor)
    end
    println("AugLag iterations: $(info.iter) with $np ranks in $elapsed_t seconds")
    @show(info.maxviol_t)
    @show(info.maxviol_t_actual)
    @show(info.maxviol_d)
    @show(info.objvalue/modelinfo.obj_scale)
    @show(info.lyapunov/modelinfo.obj_scale)
else
    info = ProxAL.optimize!(nlp; ρ_t_initial = ρ_t_initial, τ_factor = τ_factor)
end

MPI.Finalize()
