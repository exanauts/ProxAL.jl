using Test
using ProxAL
using DelimitedFiles, Printf
using LinearAlgebra, JuMP
using CatViews
using CUDA
using MPI

use_MPI = true

use_MPI && MPI.Init()
DATA_DIR = joinpath(dirname(@__FILE__), "..", "data")
rtol = 1e-4

#(case, T, K, ramp_scale, load_scale, corr_scale) = ("case9241pegase", 96, 0, 0.3, 0.8, 0.5)
(case, T, K, ramp_scale, load_scale, corr_scale) = ("case118", 10, 3, 0.2, 1.0, 0.5)
#(case, T, K, ramp_scale, load_scale, corr_scale) = ("case30", 10, 5, 0.2, 1.0, 0.5)

# Load case
case_file = joinpath(DATA_DIR, "$(case).m")
load_file = joinpath(DATA_DIR, "mp_demand", "$(case)_oneweek_168_debug")

# Model/formulation settings
modelinfo = ModelInfo()
modelinfo.case_name = case
modelinfo.num_time_periods = T
modelinfo.num_ctgs = K
modelinfo.load_scale = load_scale
modelinfo.ramp_scale = ramp_scale
modelinfo.corr_scale = corr_scale
modelinfo.allow_obj_gencost = true
modelinfo.allow_constr_infeas = false
modelinfo.weight_constr_infeas = 1e9
modelinfo.time_link_constr_type = :penalty
modelinfo.ctgs_link_constr_type = :corrective_penalty
modelinfo.weight_freq_ctrl = 1e6
modelinfo.obj_scale = 1e-3
modelinfo.allow_line_limits = false

# Algorithm settings
algparams = AlgParams()
algparams.verbose = 1
algparams.mode = :coldstart
algparams.tol = 1e-3

solver = "Ipopt"
using Ipopt
algparams.optimizer =
    optimizer_with_attributes(Ipopt.Optimizer, #"tol" => 1e-1*algparams.tol,
        "print_level" => 0)


        #=
algparams.mode = :nondecomposed
algparams.θ_t = algparams.θ_c = (1/algparams.tol)^2
nlp = NonDecomposedModel(case_file, load_file, modelinfo, algparams)
result = ProxAL.optimize!(nlp)
# @test isapprox(result["objective_value_nondecomposed"], OPTIMAL_OBJVALUE, rtol = rtol)
# @test isapprox(result["primal"].Pg[:], OPTIMAL_PG, rtol = rtol)
# @test norm(result["primal"].Zt[:], Inf) <= algparams.tol
=#


algparams.mode = :coldstart
algparams.decompCtgs = true
algparams.iterlim = 10
algparams.nlpiterlim = 1000


#=
nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, ProxAL.JuMPBackend(), Dict(), Dict(), nothing)
runinfo = ProxAL.optimize!(nlp);
# @show(result["primal"].Zk[:])
# @show(runinfo.x.Zk[:])
# @show(norm(result["primal"].sigma_imag[:]))
# @show(norm(result["primal"].sigma_real[:]))
# @show(norm(result["primal"].sigma_lineFr[:]))
# @show(norm(result["primal"].sigma_lineTo[:]))
# @test isapprox(result["primal"].ωt[:], runinfo.x.ωt[:], rtol = 1e-1)
# @test isapprox(result["primal"].Pg[:], runinfo.x.Pg[:], rtol = 1e-2)
=#
# algparams.updateρ_t = false
# algparams.updateτ = false

nlp = use_MPI ? ProxALEvaluator(case_file, load_file, modelinfo, algparams, ProxAL.JuMPBackend(), Dict(), Dict()) :
                ProxALEvaluator(case_file, load_file, modelinfo, algparams, ProxAL.JuMPBackend(), Dict(), Dict(), nothing)
if use_MPI && MPI.Comm_rank(MPI.COMM_WORLD) == 0
    np = MPI.Comm_size(MPI.COMM_WORLD)
    elapsed_t = @elapsed begin
      runinfo = ProxAL.optimize!(nlp);
    end
    println("AugLag iterations: $(runinfo.iter) with $np ranks in $elapsed_t seconds")
else
  runinfo = ProxAL.optimize!(nlp);
end

use_MPI && MPI.Finalize()
!use_MPI && println("done")
