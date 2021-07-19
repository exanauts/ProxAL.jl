using Test
using ProxAL
using DelimitedFiles, Printf
using LinearAlgebra, JuMP
using CatViews
using CUDA

DATA_DIR = joinpath(dirname(@__FILE__), "..", "data")
case = "case9"
T = 2
ramp_scale = 0.5
load_scale = 1.0
maxρ = 0.1
quad_penalty = 0.1
rtol = 1e-4

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

# Algorithm settings
algparams = AlgParams()
algparams.verbose = 0
algparams.θ_t = algparams.θ_c = quad_penalty
algparams.ρ_t = algparams.ρ_c = maxρ
algparams.τ = 3maxρ

solver_list = ["Ipopt"]
# TODO: MadNLP broken currently
# solver_list = ["Ipopt", "MadNLP"]
# if CUDA.has_cuda_gpu()
#     push!(solver_list, "MadNLPGPU")
# end
if isfile(joinpath(dirname(@__FILE__), "..", "build/libhiop.so"))
    push!(solver_list, "Hiop")
    ENV["JULIA_HIOP_LIBRARY_PATH"] = joinpath(dirname(@__FILE__), "..", "build")
    @info("Using Hiop at $(ENV["JULIA_HIOP_LIBRARY_PATH"])")
end

@testset "Test ProxAL on $(case)" begin
    modelinfo.case_name = case

    for solver in solver_list
    solver = "Ipopt"
    @testset "$(solver)" begin
        println("Testing using $(solver)")
        if solver == "Ipopt"
            using Ipopt
            algparams.optimizer =
                optimizer_with_attributes(Ipopt.Optimizer,
                    "print_level" => Int64(algparams.verbose > 0)*5)
        end


        @testset "solver = $(solver)" begin

            K = 0
            algparams.decompCtgs = false
            @testset "$T-period, $K-ctgs, time_link=penalty" begin
                modelinfo.num_ctgs = K
                optimal_obvalue = round(11.258316111585623, digits = 6)
                optimal_pg = round.([0.8979870694509675, 1.3432060120295906, 0.9418738103137331, 0.9840203268625166, 1.448040098924617, 1.0149638876964715], digits = 5)

                @testset "Non-decomposed formulation" begin
                    algparams.mode = :nondecomposed
                    algparams.ρ_t = algparams.ρ_c = maxρ
                    algparams.τ = 3maxρ
                    nlp = NonDecomposedModel(case_file, load_file, modelinfo, algparams)
                    result = ProxAL.optimize!(nlp)
                    @test isapprox(result["objective_value_nondecomposed"], optimal_obvalue, rtol = rtol)
                    @test isapprox(result["primal"].Pg[:], optimal_pg, rtol = rtol)
                    @test norm(result["primal"].Zt[:], Inf) <= 1e-4
                end

                @testset "Lyapunov bound" begin
                    algparams.mode = :lyapunov_bound
                    algparams.ρ_t = algparams.ρ_c = maxρ
                    algparams.τ = 3maxρ
                    nlp = NonDecomposedModel(case_file, load_file, modelinfo, algparams)
                    result = ProxAL.optimize!(nlp)
                    @test isapprox(result["objective_value_lyapunov_bound"], optimal_obvalue)
                    @test isapprox(result["primal"].Pg[:], optimal_pg, rtol = rtol)
                end

                @testset "ProxALM" begin
                    algparams.mode = :coldstart
                    algparams.ρ_t = algparams.ρ_c = maxρ
                    algparams.τ = 3maxρ
                    nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, JuMPBackend(), Dict(), Dict(), nothing)
                    runinfo = ProxAL.optimize!(nlp)
                    @test isapprox(runinfo.objvalue[end], optimal_obvalue, rtol = rtol)
                    @test isapprox(runinfo.x.Pg[:], optimal_pg, rtol = rtol)
                    @test norm(runinfo.x.Zt[:], Inf) <= 1e-4
                    @test isapprox(runinfo.maxviol_c[end], 0.0)
                    @test runinfo.maxviol_t[end] <= algparams.tol
                    @test runinfo.maxviol_d[end] <= algparams.tol
                    @test runinfo.iter <= algparams.iterlim
                end
            end



            K = 1
            algparams.decompCtgs = false
            @testset "$T-period, $K-ctgs, time_link=penalty, ctgs_link=frequency_ctrl" begin
                modelinfo.num_ctgs = K
                optimal_obvalue = round(11.258316111574212, digits = 6)
                optimal_pg = round.([0.8979870693416382, 1.3432060108971793, 0.9418738115511179, 0.9055318507524525, 1.3522597485901564, 0.9500221754747974, 0.9840203265549852, 1.4480400977338292, 1.014963889201792, 0.9932006221514175, 1.459056452449548, 1.024878608445939], digits = 5)
                optimal_ωt = round.([0.0, -0.00012071650257302939, 0.0, -0.00014688472954291597], sigdigits = 4)

                @testset "Non-decomposed formulation" begin
                    algparams.mode = :nondecomposed
                    algparams.ρ_t = algparams.ρ_c = maxρ
                    algparams.τ = 3maxρ
                    nlp = NonDecomposedModel(case_file, load_file, modelinfo, algparams)
                    result = ProxAL.optimize!(nlp)
                    @test isapprox(result["objective_value_nondecomposed"], optimal_obvalue, rtol = rtol)
                    @test isapprox(result["primal"].Pg[:], optimal_pg, rtol = rtol)
                    @test norm(result["primal"].Zt[:], Inf) <= 1e-4
                    @test isapprox(result["primal"].ωt[:], optimal_ωt, rtol = 1e-1)
                end
                @testset "Lyapunov bound" begin
                    algparams.mode = :lyapunov_bound
                    algparams.ρ_t = algparams.ρ_c = maxρ
                    algparams.τ = 3maxρ
                    nlp = NonDecomposedModel(case_file, load_file, modelinfo, algparams)
                    result = ProxAL.optimize!(nlp)
                    @test isapprox(result["objective_value_lyapunov_bound"], optimal_obvalue)
                    @test isapprox(result["primal"].Pg[:], optimal_pg, rtol = rtol) # [0.8979870693416395, 1.3432060108971777, 0.9418738115511173, 0.9055318507524539, 1.3522597485901549, 0.9500221754747968, 0.9840203265549855, 1.4480400977338292, 1.0149638892017916, 0.9932006221514178, 1.459056452449548, 1.0248786084459387]
                end

                @testset "ProxALM" begin
                    algparams.mode = :coldstart
                    algparams.ρ_t = algparams.ρ_c = maxρ
                    algparams.τ = 3maxρ
                    nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, JuMPBackend(), Dict(), Dict(), nothing)
                    runinfo = ProxAL.optimize!(nlp)
                    @test isapprox(runinfo.objvalue[end], optimal_obvalue, rtol = rtol)
                    @test isapprox(runinfo.x.Pg[:], optimal_pg, rtol = rtol)
                    @test norm(runinfo.x.Zt[:], Inf) <= 1e-4
                    @test isapprox(runinfo.x.ωt[:], optimal_ωt, rtol = 1e-1)
                    @test isapprox(runinfo.maxviol_c[end], 0.0)
                    @test runinfo.maxviol_t[end] <= algparams.tol
                    @test runinfo.maxviol_d[end] <= algparams.tol
                    @test runinfo.iter <= algparams.iterlim
                end
            end




            K = 1
            algparams.decompCtgs = true
            @testset "$T-period, $K-ctgs, time_link=penalty, ctgs_link=frequency_ctrl, decompCtgs" begin
                modelinfo.num_ctgs = K
                optimal_obvalue = round(11.258316111574212, digits = 6)
                optimal_pg = round.([0.8979870693416382, 1.3432060108971793, 0.9418738115511179, 0.9055318507524525, 1.3522597485901564, 0.9500221754747974, 0.9840203265549852, 1.4480400977338292, 1.014963889201792, 0.9932006221514175, 1.459056452449548, 1.024878608445939], digits = 5)
                optimal_ωt = round.([0.0, -0.00012071650257302939, 0.0, -0.00014688472954291597], sigdigits = 4)

                @testset "Non-decomposed formulation" begin
                    algparams.mode = :nondecomposed
                    algparams.ρ_t = algparams.ρ_c = maxρ
                    algparams.τ = 3maxρ
                    nlp = NonDecomposedModel(case_file, load_file, modelinfo, algparams)
                    result = ProxAL.optimize!(nlp)
                    @test isapprox(result["objective_value_nondecomposed"], optimal_obvalue, rtol = rtol)
                    @test isapprox(result["primal"].Pg[:], optimal_pg, rtol = rtol)
                    @test norm(result["primal"].Zt[:], Inf) <= 1e-4
                    @test isapprox(result["primal"].ωt[:], optimal_ωt, rtol = 1e-1)
                end
                @testset "Lyapunov bound" begin
                    algparams.mode = :lyapunov_bound
                    algparams.ρ_t = algparams.ρ_c = maxρ
                    algparams.τ = 3maxρ
                    nlp = NonDecomposedModel(case_file, load_file, modelinfo, algparams)
                    result = ProxAL.optimize!(nlp)
                    @test isapprox(result["objective_value_lyapunov_bound"], optimal_obvalue)
                    @test isapprox(result["primal"].Pg[:], optimal_pg, rtol = rtol)
                end
                @testset "ProxALM" begin
                    algparams.mode = :coldstart
                    algparams.ρ_t = algparams.ρ_c = maxρ
                    algparams.τ = 3maxρ
                    nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, JuMPBackend(), Dict(), Dict(), nothing)
                    runinfo = ProxAL.optimize!(nlp)
                    @test isapprox(runinfo.objvalue[end], optimal_obvalue, rtol = rtol)
                    @test_broken isapprox(runinfo.x.Pg[:], optimal_pg, rtol = rtol)
                    @test_broken norm(runinfo.x.Zt[:], Inf) <= 1e-4
                    @test_broken isapprox(runinfo.x.ωt[:], optimal_ωt, rtol = 1e-1)
                    @test runinfo.maxviol_c[end] <= algparams.tol
                    @test runinfo.maxviol_t[end] <= algparams.tol
                    @test runinfo.maxviol_d[end] <= algparams.tol
                    @test runinfo.iter <= algparams.iterlim
                end
            end
        end
    end # solver testset
    end
end
