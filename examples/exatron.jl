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
using Enzyme
using Enzyme_jll

MPI.Init()

# Select one of the following
# (case, T, ramp_scale, load_scale, maxρ, quad_penalty) = ("case_ACTIVSg2000_Corrected", 20, 0.3, 1.0, 0.1, 1e5) # 290 iterations
# (case, T, ramp_scale, load_scale, maxρ, quad_penalty) = ("case_ACTIVSg2000_Corrected", 20, 0.3, 1.0, 0.01, 1e3) # 449 iterations
# (case, T, ramp_scale, load_scale, maxρ, quad_penalty) = ("case_ACTIVSg2000_Corrected", 20, 0.3, 1.0, 0.01, 1e5) # 521 iterations
# (case, T, ramp_scale, load_scale, maxρ, quad_penalty) = ("case_ACTIVSg2000_Corrected", 20, 0.3, 1.0, 0.1, 1e3) # 225 iterations
# (case, T, ramp_scale, load_scale, maxρ, quad_penalty) = ("case9241pegase", 2, 0.3, 0.8, 0.01, 1e3)
# (case, T, ramp_scale, load_scale, maxρ, quad_penalty) = ("case118", 168, 0.2, 1.0, 0.1, 1e5)
(case, T, ramp_scale, load_scale, maxρ, quad_penalty) = ("case9", 2, 0.04, 1.0, 0.1, 0.1)

T = parse(Int, ARGS[1])
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
  println("$T periods")
end

# No contingencies in this example
K = 0

# Load case
DATA_DIR = joinpath(dirname(@__FILE__), "..", "data")
case_file = joinpath(DATA_DIR, "$(case).m")
load_file = joinpath(DATA_DIR, "mp_demand", "$(case)_oneweek_168")
# load_file = joinpath(DATA_DIR, "mp_demand", "$(case)_3600")

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
algparams.verbose = 0
algparams.tol = 1e-3
algparams.decompCtgs = true
algparams.iterlim = 1
algparams.device = ProxAL.CUDADevice
algparams.optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
algparams.tron_rho_pq=5*1e4
algparams.tron_rho_pa=5*1e5
algparams.tron_outer_iterlim=30
algparams.tron_inner_iterlim=2000
algparams.tron_scale=1e-5
algparams.mode = :coldstart

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

# if MPI.Comm_rank(MPI.COMM_WORLD) == 0
function f(x,y)
    nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, ProxAL.ExaTronBackend(), opt_sol, lyapunov_sol, nothing)

    nlp.alminfo.x.Pg[1] = x[1]
    info = ProxAL.optimize!(nlp)
    y[1] = info.x.Pg[1]
    nothing
end

x  = Float64[1.3]
dx = Float64[0.0]
y  = Float64[0.0]
dy = Float64[1.0]
println("Before passive run: x=$x, y=$y")
f(x,y)
println("Before autodiff run: x=$x, dx=$dx, y=$y, dy=$dy")
autodiff(f, Duplicated(x,dx), Duplicated(y,dy))
println("After autodiff run: x=$x, dx=$dx, y=$y, dy=$dy")

# else
#   info = ProxAL.optimize!(nlp)
# end

MPI.Finalize()
