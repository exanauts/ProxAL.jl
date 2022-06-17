using Test
using ProxAL
using DelimitedFiles, Printf
using LinearAlgebra, JuMP, Ipopt
using CatViews
using CUDA
using MPI
using LazyArtifacts
is_my_work(blk, comm::MPI.Comm) = (blk % MPI.Comm_size(comm)) == MPI.Comm_rank(comm)

use_MPI = !isempty(ARGS) && (parse(Bool, ARGS[1]) == true)
use_MPI && MPI.Init()
const DATA_DIR = joinpath(artifact"ExaData", "ExaData")
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
modelinfo.weight_freq_ctrl = 1e7
modelinfo.time_link_constr_type = :frequency_recovery
modelinfo.obj_scale = 1e-4

# Algorithm settings
algparams = AlgParams()
algparams.verbose = 0
algparams.optimizer =
                optimizer_with_attributes(Ipopt.Optimizer,
                    "print_level" => 0)

solver_list = ["Ipopt", "ExaAdmmCPU"]
if CUDA.has_cuda_gpu()
    # TODO: MadNLP broken currently
    # push!(solver_list, "MadNLPGPU")
    # push!(solver_list, "ExaAdmmGPU")
end

@testset "Test ProxAL on $(case)" begin
    modelinfo.case_name = case

    for solver in solver_list
    @testset "$(solver)" begin
        println("Testing using $(solver)")
        if solver == "Ipopt"
            backend = JuMPBackend()
        end
        if solver == "ExaAdmmCPU"
            backend = AdmmBackend()
            algparams.tron_outer_iterlim=2000
            algparams.tron_outer_eps=1e-6
        end
        if solver == "ExaAdmmGPU"
            backend = AdmmBackend()
            algparams.tron_outer_iterlim=2000
            algparams.tron_outer_eps=1e-6
            algparams.device = ProxAL.CUDADevice
        end


        @testset "solver = $(solver)" begin

            
            K = 0
            algparams.decompCtgs = false
            @testset "$T-period, $K-ctgs, time_link=$(modelinfo.time_link_constr_type)" begin
                modelinfo.num_ctgs = K
                OPTIMAL_OBJVALUE = 11258.31609659974*modelinfo.obj_scale
                OPTIMAL_PG = round.([0.8979870696290546, 1.3432060116663056, 0.9418738104926565, 0.9840203270260095, 1.4480400985949127, 1.014963887857111], digits = 5)

                if solver == "Ipopt"
                    @testset "Non-decomposed formulation" begin
                        algparams.mode = :nondecomposed
                        algparams.θ_t = algparams.θ_c = (1/algparams.tol)^2
                        nlp = NonDecomposedModel(case_file, load_file, modelinfo, algparams)
                        runinfo = ProxAL.optimize!(nlp)
                        @test isapprox(runinfo.objvalue[end], OPTIMAL_OBJVALUE, rtol = rtol)
                        @test isapprox(runinfo.x.Pg[:], OPTIMAL_PG, rtol = rtol)
                        @test isapprox(runinfo.x.Pr[:], OPTIMAL_PG, rtol = rtol)
                        @test norm(runinfo.x.ωt[:], Inf) <= 1e-4
                        @test norm(runinfo.x.Zt[:], Inf) <= algparams.tol
                    end
                end

                @testset "ProxAL" begin
                    algparams.mode = :coldstart
                    nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, backend, use_MPI ? MPI.COMM_WORLD : nothing)
                    runinfo = ProxAL.optimize!(nlp)
                    @test isapprox(runinfo.objvalue[end], OPTIMAL_OBJVALUE, rtol = rtol)
                    if !use_MPI || is_my_work(1, MPI.COMM_WORLD)
                        @test isapprox(runinfo.x.Pg[:], OPTIMAL_PG, rtol = rtol)
                        @test isapprox(runinfo.x.Pr[:], OPTIMAL_PG, rtol = 10.0*rtol)
                        @test norm(runinfo.x.ωt[:], Inf) <= 1e-4
                    end
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
                    proxal_ctgs_link = :frequency_penalty
                elseif ctgs_link == :preventive_equality
                    OPTIMAL_OBJVALUE = round(11340.093581315918*modelinfo.obj_scale, digits = 6)
                    OPTIMAL_PG = round.([0.8802072707975164, 1.3551904372529155, 0.9621162845248026, 0.9632463589831285, 1.46270646294457, 1.03946802455576], digits = 5)
                    proxal_ctgs_link = :preventive_penalty
                elseif ctgs_link == :corrective_inequality
                    OPTIMAL_OBJVALUE = round(11258.316096599581*modelinfo.obj_scale, digits = 6)
                    OPTIMAL_PG = round.([0.8979870771538515, 1.3432060071606775, 0.941873804023475, 0.9840203279201325, 1.4480400975658037, 1.0149638851762735], digits = 5)
                    proxal_ctgs_link = :corrective_penalty
                end
                if contains(solver, "ExaAdmm") && proxal_ctgs_link == :frequency_penalty
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
                            runinfo = ProxAL.optimize!(nlp)
                            @test isapprox(runinfo.objvalue[end], OPTIMAL_OBJVALUE, rtol = rtol)
                            @test isapprox(runinfo.x.Pg[:,1,:][:], OPTIMAL_PG, rtol = rtol)
                            @test isapprox(runinfo.x.Pr[:], OPTIMAL_PG, rtol = rtol)
                            @test norm(runinfo.x.ωt[:], Inf) <= 1e-4
                            @test norm(runinfo.x.Zt[:], Inf) <= algparams.tol
                        end

                        @testset "ProxAL" begin
                            algparams.mode = :coldstart
                            nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, backend, use_MPI ? MPI.COMM_WORLD : nothing)
                            runinfo = ProxAL.optimize!(nlp)
                            @test isapprox(runinfo.objvalue[end], OPTIMAL_OBJVALUE, rtol = rtol)
                            if !use_MPI || is_my_work(1, MPI.COMM_WORLD)
                                @test isapprox(runinfo.x.Pg[:,1,:][:], OPTIMAL_PG, rtol = rtol)
                                @test isapprox(runinfo.x.Pr[:], OPTIMAL_PG, rtol = 100.0*rtol)
                            end
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
                            runinfo = ProxAL.optimize!(nlp)
                            @test isapprox(runinfo.objvalue[end], OPTIMAL_OBJVALUE, rtol = rtol)
                            @test isapprox(runinfo.x.Pg[:,1,:][:], OPTIMAL_PG, rtol = rtol)
                            @test isapprox(runinfo.x.Pr[:], OPTIMAL_PG, rtol = rtol)
                            @test norm(runinfo.x.ωt[:], Inf) <= 1e-4
                            @test norm(runinfo.x.Zt[:], Inf) <= algparams.tol
                            @test norm(runinfo.x.Zk[:], Inf) <= algparams.tol
                        end
                    end

                    @testset "ProxAL" begin
                        algparams.mode = :coldstart
                        algparams.iterlim = 300
                        nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, backend, use_MPI ? MPI.COMM_WORLD : nothing)
                        runinfo = ProxAL.optimize!(nlp)
                        @test isapprox(runinfo.objvalue[end], OPTIMAL_OBJVALUE, rtol = 1e-2)
                        if !use_MPI || is_my_work(1, MPI.COMM_WORLD)
                            @test isapprox(runinfo.x.Pg[:,1,:][:], OPTIMAL_PG, rtol = 1e-2)
                            @test isapprox(runinfo.x.Pr[:], OPTIMAL_PG, rtol = 100.0*rtol)
                            @test norm(runinfo.x.ωt[:], Inf) <= 1e-4
                        end
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

use_MPI && MPI.Finalize()
