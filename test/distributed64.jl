using ProxAL
using DelimitedFiles, Printf
using LinearAlgebra, JuMP, Ipopt
using CatViews
using CUDA
using MPI
using MadNLP

MPI.Init()
DATA_DIR = joinpath(dirname(@__FILE__), "..", "data")
case = "case9"
T = 64
K = 0
ramp_scale = 0.04
load_scale = 1.0
maxρ = 0.1
quad_penalty = 0.1

# Select one of the following
(case, T, ramp_scale, load_scale, maxρ, quad_penalty) = ("case9241pegase", 168, 0.3, 0.8, 0.01, 1e3)
(case, T, ramp_scale, load_scale, maxρ, quad_penalty) = ("case118", 168, 0.2, 1.0, 0.1, 1e5)
(case, T, ramp_scale, load_scale, maxρ, quad_penalty) = ("case9", 64, 0.04, 1.0, 0.1, 0.1)

# Load case
case_file = joinpath(DATA_DIR, "$(case).m")
load_file = joinpath(DATA_DIR, "mp_demand", "$(case)_oneweek_168")
# ctgs_arr = deepcopy(rawdata.ctgs_arr)

# Model/formulation settings
modelinfo = ModelParams()
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
algparams.parallel = true #algparams.parallel = (nprocs() > 1)
algparams.verbose = 0
algparams.θ_t = quad_penalty
algparams.ρ_t = algparams.ρ_c = maxρ
algparams.τ = 3maxρ
algparams.decompCtgs = false
algparams.optimizer =
optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
# algparams.optimizer = () ->
#     MadNLP.Optimizer(linear_solver="LapackGPU",
#                         print_level=MadNLP.ERROR,
#                         max_iter=300)


# rawdata.ctgs_arr = deepcopy(ctgs_arr[1:modelinfo.num_ctgs])

if case in ["case9",  "case118"]
    algparams.mode = :nondecomposed
    nondecomposed = NonDecomposedModel(case_file, load_file, modelinfo, algparams)
    opt_sol = ProxAL.optimize!(nondecomposed)
    zstar = opt_sol["objective_value_nondecomposed"]

    algparams.mode = :lyapunov_bound
    nondecomposed = NonDecomposedModel(case_file, load_file, modelinfo, algparams)
    lyapunov_sol = ProxAL.optimize!(nondecomposed)
    lyapunov_star = lyapunov_sol["objective_value_lyapunov_bound"]
else
    opt_sol, lyapunov_sol = Dict(), Dict()
    zstar, lyapunov_star = NaN, NaN
end

algparams.mode = :coldstart
nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, ProxAL.FullSpace(), opt_sol, lyapunov_sol)
info = ProxAL.optimize!(nlp)
nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, ProxAL.FullSpace(), opt_sol, lyapunov_sol)
info = ProxAL.optimize!(nlp)
nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, ProxAL.FullSpace(), opt_sol, lyapunov_sol)
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    np = MPI.Comm_size(MPI.COMM_WORLD)
    @time info = ProxAL.optimize!(nlp)
    println("AugLag iterations: $(info.iter) with $np ranks")
    optimgap = 100.0 * abs.(info.objvalue .- zstar)/abs(zstar)
    lyapunov_gap = 100.0 * (info.lyapunov .- lyapunov_star)/abs(lyapunov_star)
    lyapunov_gap[lyapunov_gap .< 0] .= NaN

    @show info.iter
    @show info.maxviol_t
    @show info.maxviol_d
    @show info.dist_x
    @show optimgap
    @show lyapunov_gap
else
  info = ProxAL.optimize!(nlp)
end


MPI.Finalize()

