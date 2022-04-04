#############################################################################
# Test scripts for ACTIVSg2000 and ACTIVSg10k
#############################################################################

using ProxAL
using DelimitedFiles, Printf
using LinearAlgebra, JuMP, Ipopt
using CatViews
using Logging
using LazyArtifacts
using MPI, CUDA

MPI.Init()

solver     = length(ARGS) > 0 ? ARGS[1] : "exatron"
num_sweeps = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 2
rho0       = length(ARGS) > 2 ? parse(Float64, ARGS[3]) : 1e-3
obj_scale  = length(ARGS) > 3 ? parse(Float64, ARGS[4]) : 1e-3

# choose case
case = "case118"
case = "case_ACTIVSg10k"
resolution = 30 #in minutes
summit = true

# choose backend
backend = (solver == "ipopt") ? ProxAL.JuMPBackend() : ProxAL.ExaTronBackend()

# case file
case_file = joinpath(artifact"ExaData", "ExaData/matpower/$(case).m")

# Model/formulation settings
modelinfo = ModelInfo()
modelinfo.case_name = case
modelinfo.num_time_periods = 12
modelinfo.load_scale = 1.0
if startswith(case, "case_ACTIVSg")
    tfile = Int(24 * 7 * (60 / resolution))
    load_file = summit ? "/gpfs/alpine/proj-shared/csc359" : "/scratch"
    load_file = joinpath(load_file, "ACTIVSg_Time_Series/mp_demand/$(case)_Jun_oneweek_$(tfile)_$(resolution)min")
    modelinfo.ramp_scale = Float64(resolution)
else
    load_file = joinpath(artifact"ExaData", "ExaData", "mp_demand", "$(case)_oneweek_168")
    modelinfo.ramp_scale = 0.3
end
modelinfo.corr_scale = 0.5
modelinfo.allow_obj_gencost = true
if isa(backend, ProxAL.ExaTronBackend)
    modelinfo.allow_constr_infeas = false
else
    modelinfo.allow_constr_infeas = true
    modelinfo.weight_constr_infeas = 1e4
end
modelinfo.time_link_constr_type = :penalty
modelinfo.ctgs_link_constr_type = :corrective_penalty
modelinfo.allow_line_limits = false
modelinfo.num_ctgs = 0
modelinfo.obj_scale = obj_scale

# Algorithm settings
algparams = AlgParams()
algparams.verbose = 1
algparams.tol = 1e-3
algparams.decompCtgs = false
algparams.iterlim = (solver == "ipopt") ? 100 : 500
algparams.tron_rho_pq = 4e3
algparams.tron_rho_pa = 4e4
algparams.tron_outer_iterlim = 30
algparams.tron_inner_iterlim = 1000
algparams.tron_scale = 1e-4
algparams.num_sweeps = num_sweeps
if isa(backend, ProxAL.ExaTronBackend)
    algparams.device = ProxAL.CUDADevice
end
algparams.optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0, "tol" => 1e-6) # "linear_solver" => "ma27"
algparams.init_opf = false

# Set up and solve problem
algparams.mode = :coldstart
ranks = MPI.Comm_size(MPI.COMM_WORLD)
cur_logger = global_logger(NullLogger())
elapsed_t = @elapsed begin
    redirect_stdout(devnull) do
        global nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, backend)
    end
end
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    global_logger(cur_logger)
    println("Creating problem: $elapsed_t")
    np = MPI.Comm_size(MPI.COMM_WORLD)
    elapsed_t = @elapsed begin
        info = ProxAL.optimize!(nlp; ρ_t_initial = rho0)
    end

    @show(info.iter)
    @show(info.maxviol_d)
    @show(info.maxviol_t_actual)
    @show(info.objvalue)
    @show(info.wall_time_elapsed_actual)
    @show(info.wall_time_elapsed_ideal)
else
    info = ProxAL.optimize!(nlp; ρ_t_initial = rho0)
end

MPI.Finalize()
