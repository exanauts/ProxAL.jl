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
rtol = 1e-4

# Load case
case_file = joinpath(DATA_DIR, "$(case).m")
load_file = joinpath(DATA_DIR, "mp_demand", "$(case)_oneweek_168")

# Model/formulation settings
modelinfo = ModelInfo()
modelinfo.num_time_periods = T
modelinfo.load_scale = load_scale
modelinfo.ramp_scale = ramp_scale
modelinfo.allow_obj_gencost = true
modelinfo.allow_constr_infeas = false
modelinfo.weight_freq_ctrl = 0.1
modelinfo.time_link_constr_type = :penalty

# Algorithm settings
algparams = AlgParams()
algparams.verbose = 0

solver_list = ["Ipopt", "ExaTron"]
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
    @testset "$(solver)" begin
        println("Testing using $(solver)")
        if solver == "Ipopt"
            using Ipopt
            backend = JuMPBackend()
            algparams.optimizer =
                optimizer_with_attributes(Ipopt.Optimizer,
                    "print_level" => Int64(algparams.verbose > 0)*5)
        end
        if solver == "ExaTron"
            using ExaTron
            backend = ExaTronBackend()
            algparams.tron_outer_iterlim=2000
            algparams.tron_outer_eps=1e-6
        end


        @testset "solver = $(solver)" begin

            K = 0
            algparams.decompCtgs = false
            @testset "$T-period, $K-ctgs, time_link=$(modelinfo.time_link_constr_type)" begin
                modelinfo.num_ctgs = K
                OPTIMAL_OBJVALUE = round(11258.316096599736*modelinfo.obj_scale, digits = 6)
                OPTIMAL_PG = round.([0.8979870694509675, 1.3432060120295906, 0.9418738103137331, 0.9840203268625166, 1.448040098924617, 1.0149638876964715], digits = 5)

                if solver == "Ipopt"
                    @testset "Non-decomposed formulation" begin
                        algparams.mode = :nondecomposed
                        algparams.θ_t = algparams.θ_c = (1/algparams.tol)^2
                        nlp = NonDecomposedModel(case_file, load_file, modelinfo, algparams)
                        result = ProxAL.optimize!(nlp)
                        @test isapprox(result["objective_value_nondecomposed"], OPTIMAL_OBJVALUE, rtol = rtol)
                        @test isapprox(result["primal"].Pg[:], OPTIMAL_PG, rtol = rtol)
                        @test norm(result["primal"].Zt[:], Inf) <= algparams.tol
                    end
                end

                @testset "ProxAL" begin
                    algparams.mode = :coldstart
                    nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, backend, Dict(), Dict(), nothing)
                    runinfo = ProxAL.optimize!(nlp)
                    @test isapprox(runinfo.objvalue[end], OPTIMAL_OBJVALUE, rtol = rtol)
                    @test isapprox(runinfo.x.Pg[:], OPTIMAL_PG, rtol = rtol)
                    @test isapprox(runinfo.maxviol_c[end], 0.0)
                    @test isapprox(runinfo.maxviol_c_actual[end], 0.0)
                    @test runinfo.maxviol_t[end] <= algparams.tol
                    @test runinfo.maxviol_t_actual[end] <= algparams.tol
                    @test runinfo.maxviol_d[end] <= algparams.tol
                    @test runinfo.iter <= algparams.iterlim
                end
            end



            K = 1
            modelinfo.num_ctgs = K
            for ctgs_link in [:frequency_equality, :preventive_equality, :corrective_inequality]
                if ctgs_link == :frequency_equality
                    OPTIMAL_OBJVALUE = round(11258.316096601551*modelinfo.obj_scale, digits = 6)
                    OPTIMAL_PG = round.([0.8979870771416967, 1.3432060071608862, 0.9418738040358301, 0.9840203279072699, 1.448040097565594, 1.0149638851897345], digits = 5)
                    OPTIMAL_WT = round.([0.0, -0.0001206958510371199, 0.0, -0.0001468516901269684], sigdigits = 4)
                    proxal_ctgs_link = :frequency_penalty
                elseif ctgs_link == :preventive_equality
                    OPTIMAL_OBJVALUE = round(11340.093581315918*modelinfo.obj_scale, digits = 6)
                    OPTIMAL_PG = round.([0.8802072707975164, 1.3551904372529155, 0.9621162845248026, 0.9632463589831285, 1.46270646294457, 1.03946802455576], digits = 5)
                    OPTIMAL_WT = [0.0, 0.0, 0.0, 0.0]
                    proxal_ctgs_link = :preventive_penalty
                elseif ctgs_link == :corrective_inequality
                    OPTIMAL_OBJVALUE = round(11258.316096599581*modelinfo.obj_scale, digits = 6)
                    OPTIMAL_PG = round.([0.8979870771538515, 1.3432060071606775, 0.941873804023475, 0.9840203279201325, 1.4480400975658037, 1.0149638851762735], digits = 5)
                    OPTIMAL_WT = [0.0, 0.0, 0.0, 0.0]
                    proxal_ctgs_link = :corrective_penalty
                end
                if solver == "ExaTron" && proxal_ctgs_link != :corrective_penalty
                    continue
                end

                algparams.decompCtgs = false
                modelinfo.ctgs_link_constr_type = ctgs_link
                if solver == "Ipopt"
                    @testset "$T-period, $K-ctgs, time_link=$(modelinfo.time_link_constr_type), ctgs_link=$(ctgs_link)" begin
                        @testset "Non-decomposed formulation" begin
                            algparams.mode = :nondecomposed
                            algparams.θ_t = algparams.θ_c = (1/algparams.tol)^2
                            nlp = NonDecomposedModel(case_file, load_file, modelinfo, algparams)
                            result = ProxAL.optimize!(nlp)
                            @test isapprox(result["objective_value_nondecomposed"], OPTIMAL_OBJVALUE, rtol = rtol)
                            @test isapprox(result["primal"].Pg[:,1,:][:], OPTIMAL_PG, rtol = rtol)
                            @test norm(result["primal"].Zt[:], Inf) <= algparams.tol
                            @test isapprox(result["primal"].ωt[:], OPTIMAL_WT, rtol = 1e-1)
                        end

                        @testset "ProxAL" begin
                            algparams.mode = :coldstart
                            nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, backend, Dict(), Dict(), nothing)
                            runinfo = ProxAL.optimize!(nlp)
                            @test isapprox(runinfo.objvalue[end], OPTIMAL_OBJVALUE, rtol = rtol)
                            @test isapprox(runinfo.x.Pg[:,1,:][:], OPTIMAL_PG, rtol = rtol)
                            @test isapprox(runinfo.x.ωt[:], OPTIMAL_WT, rtol = 1e-1)
                            @test isapprox(runinfo.maxviol_c[end], 0.0)
                            @test isapprox(runinfo.maxviol_c_actual[end], 0.0)
                            @test runinfo.maxviol_t[end] <= algparams.tol
                            @test runinfo.maxviol_t_actual[end] <= algparams.tol
                            @test runinfo.maxviol_d[end] <= algparams.tol
                            @test runinfo.iter <= algparams.iterlim
                        end
                    end
                end



                algparams.decompCtgs = true
                modelinfo.ctgs_link_constr_type = proxal_ctgs_link
                @testset "$T-period, $K-ctgs, time_link=$(modelinfo.time_link_constr_type), ctgs_link=$(proxal_ctgs_link), decompCtgs" begin
                    if solver == "Ipopt"
                        @testset "Non-decomposed formulation" begin
                            algparams.mode = :nondecomposed
                            algparams.θ_t = algparams.θ_c = (10/algparams.tol)^2
                            nlp = NonDecomposedModel(case_file, load_file, modelinfo, algparams)
                            result = ProxAL.optimize!(nlp)
                            @test isapprox(result["objective_value_nondecomposed"], OPTIMAL_OBJVALUE, rtol = rtol)
                            @test isapprox(result["primal"].Pg[:,1,:][:], OPTIMAL_PG, rtol = rtol)
                            @test norm(result["primal"].Zt[:], Inf) <= algparams.tol
                            @test norm(result["primal"].Zk[:], Inf) <= algparams.tol
                        end
                    end

                    @testset "ProxAL" begin
                        algparams.mode = :coldstart
                        algparams.iterlim = 300
                        nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, backend, Dict(), Dict(), nothing)
                        runinfo = ProxAL.optimize!(nlp)
                        @test isapprox(runinfo.objvalue[end], OPTIMAL_OBJVALUE, rtol = 1e-2)
                        @test isapprox(runinfo.x.Pg[:,1,:][:], OPTIMAL_PG, rtol = 1e-2)
                        @test runinfo.maxviol_c[end] <= algparams.tol
                        @test runinfo.maxviol_t[end] <= algparams.tol
                        @test runinfo.maxviol_c_actual[end] <= algparams.tol
                        @test runinfo.maxviol_t_actual[end] <= algparams.tol
                        @test runinfo.maxviol_d[end] <= algparams.tol
                        @test runinfo.iter <= algparams.iterlim
                    end
                end
            end
        end
    end # solver testset
    end
end
