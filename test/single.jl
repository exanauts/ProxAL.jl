using Test
using ProxAL
using DelimitedFiles, Printf
using LinearAlgebra, JuMP
using CatViews
using CUDA
using MPI

MPI.Init()
DATA_DIR = joinpath(dirname(@__FILE__), "..", "data")
case = "case9"
T = 2
ramp_scale = 0.5
load_scale = 1.0
maxρ = 0.1
quad_penalty = 0.1
rtol = 1e-4

# Load case
case_file = joinpath(DATA_DIR, "$case")
load_file = joinpath(DATA_DIR, "mp_demand", "$(case)_oneweek_168")
rawdata = RawData(case_file, load_file)
ctgs_arr = deepcopy(rawdata.ctgs_arr)

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

# Algorithm settings
algparams = AlgParams()
algparams.parallel = false #algparams.parallel = (nprocs() > 1)
algparams.verbose = 0

solver_list = ["Ipopt", "MadNLP"]
if CUDA.has_cuda_gpu()
    push!(solver_list, "MadNLPGPU")
end
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
            algparams.optimizer =
                optimizer_with_attributes(Ipopt.Optimizer,
                    "print_level" => Int64(algparams.verbose > 0)*5)
        end
        if solver == "MadNLP"
            using MadNLP
            algparams.optimizer = () ->
                MadNLP.Optimizer(linear_solver="Mumps",
                                 print_level=MadNLP.ERROR,
                                 max_iter=300)
        end
        if solver == "MadNLPGPU"
            using MadNLP
            algparams.optimizer = () ->
                MadNLP.Optimizer(linear_solver="LapackCUDA",
                                 print_level=MadNLP.ERROR,
                                 max_iter=300)
        end
        if solver == "Hiop"
            using Hiop
            algparams.optimizer = optimizer_with_attributes(Hiop.Optimizer)
        end


        @testset "solver = $(solver)" begin

            K = 0
            algparams.decompCtgs = false
            @testset "$T-period, $K-ctgs, time_link=penalty" begin
                modelinfo.num_ctgs = K
                rawdata.ctgs_arr = deepcopy(ctgs_arr[1:modelinfo.num_ctgs])
                opfdata = opf_loaddata(rawdata;
                                       time_horizon_start = 1,
                                       time_horizon_end = T,
                                       load_scale = load_scale,
                                       ramp_scale = ramp_scale)
                
                set_rho!(algparams;
                         ngen = length(opfdata.generators),
                         modelinfo = modelinfo,
                         maxρ_t = maxρ,
                         maxρ_c = maxρ)
                
                @testset "Non-decomposed formulation" begin
                    algparams.mode = :nondecomposed
                    result = solve_fullmodel(opfdata, rawdata, modelinfo, algparams)
                    @test isapprox(result["objective_value_nondecomposed"], 11.258316111585623, rtol = rtol)
                    @test isapprox(result["primal"].Pg[:], [0.8979870694509675, 1.3432060120295906, 0.9418738103137331, 0.9840203268625166, 1.448040098924617, 1.0149638876964715], rtol = rtol)
                    if solver == "MadNLP" || solver == "MadNLPGPU" || solver == "Hiop"
                        @test_broken isapprox(result["primal"].Zt[:], [0.0, 0.0, 0.0, 2.7859277234613066e-6, 2.3533760802049378e-6, 2.0234235436650152e-6], rtol = rtol)
                    else
                        @test isapprox(result["primal"].Zt[:], [0.0, 0.0, 0.0, 2.7859277234613066e-6, 2.3533760802049378e-6, 2.0234235436650152e-6], rtol = rtol)
                    end
                end
                
                @testset "Lyapunov bound" begin
                    algparams.mode = :lyapunov_bound
                    result = solve_fullmodel(opfdata, rawdata, modelinfo, algparams)
                    @test isapprox(result["objective_value_lyapunov_bound"], 11.25831611158562)
                    @test isapprox(result["primal"].Pg[:], [0.8979870694509652, 1.343206012029591, 0.9418738103137334, 0.9840203268625173, 1.448040098924617, 1.014963887696471], rtol = rtol)
                end

                @testset "ProxALM" begin
                    algparams.mode = :coldstart
                    runinfo = run_proxALM(opfdata, rawdata, modelinfo, algparams)
                    @test isapprox(runinfo.maxviol_c[end], 0.0)
                    @test isapprox(runinfo.x.Pg[:], [0.8979849196165037, 1.3432106614001416, 0.9418713794662078, 0.9840203268799962, 1.4480400989162827, 1.0149638876932787], rtol = rtol)
                    @test isapprox(runinfo.λ.ramping[:], [0.0, 0.0, 0.0, 2.1600093405682597e-6, -7.2856620728201185e-6, 5.051385899057505e-6], rtol = rtol)
                    @test isapprox(runinfo.maxviol_t[end], 2.687848059435005e-5, rtol = rtol)
                    @test isapprox(runinfo.maxviol_d[end], 7.28542741650351e-6, rtol = rtol)
                    @test runinfo.iter == 5
                end
            end



            K = 1
            algparams.decompCtgs = false
            @testset "$T-period, $K-ctgs, time_link=penalty, ctgs_link=frequency_ctrl" begin
                modelinfo.num_ctgs = K
                rawdata.ctgs_arr = ctgs_arr[1:modelinfo.num_ctgs]
                opfdata = opf_loaddata(rawdata;
                                       time_horizon_start = 1,
                                       time_horizon_end = T,
                                       load_scale = load_scale,
                                       ramp_scale = ramp_scale)
                
                set_rho!(algparams;
                         ngen = length(opfdata.generators),
                         modelinfo = modelinfo,
                         maxρ_t = maxρ,
                         maxρ_c = maxρ)
            
                @testset "Non-decomposed formulation" begin
                    algparams.mode = :nondecomposed
                    result = solve_fullmodel(opfdata, rawdata, modelinfo, algparams)
                    @test isapprox(result["objective_value_nondecomposed"], 11.258316111574212, rtol = rtol)
                    if solver == "Hiop"
                        @test_broken isapprox(result["primal"].Pg[:], [0.8979870693416382, 1.3432060108971793, 0.9418738115511179, 0.9055318507524525, 1.3522597485901564, 0.9500221754747974, 0.9840203265549852, 1.4480400977338292, 1.014963889201792, 0.9932006221514175, 1.459056452449548, 1.024878608445939], rtol = rtol)
                    else
                        @test isapprox(result["primal"].Pg[:], [0.8979870693416382, 1.3432060108971793, 0.9418738115511179, 0.9055318507524525, 1.3522597485901564, 0.9500221754747974, 0.9840203265549852, 1.4480400977338292, 1.014963889201792, 0.9932006221514175, 1.459056452449548, 1.024878608445939], rtol = rtol)
                    end
                    if solver == "MadNLP" || solver == "MadNLPGPU" || solver == "Hiop"
                        @test_broken isapprox(result["primal"].Zt[:], [0.0, 0.0, 0.0, 2.7857429391709934e-6, 2.353279608425683e-6, 2.023313001658965e-6], rtol = rtol)
                    else
                        @test isapprox(result["primal"].Zt[:], [0.0, 0.0, 0.0, 2.7857429391709934e-6, 2.353279608425683e-6, 2.023313001658965e-6], rtol = rtol)
                    end
                    if solver == "Hiop"
                        @test_broken isapprox(result["primal"].ωt[:], [0.0, -0.00012071650257302939, 0.0, -0.00014688472954291597], rtol = rtol)
                    else
                        @test isapprox(result["primal"].ωt[:], [0.0, -0.00012071650257302939, 0.0, -0.00014688472954291597], rtol = rtol)
                    end
                end
                @testset "Lyapunov bound" begin
                    algparams.mode = :lyapunov_bound
                    result = solve_fullmodel(opfdata, rawdata, modelinfo, algparams)
                    if solver == "Hiop"
                        @test_broken isapprox(result["objective_value_lyapunov_bound"], 11.258316111574207)
                        @test_broken isapprox(result["primal"].Pg[:], [0.8979870693416395, 1.3432060108971777, 0.9418738115511173, 0.9055318507524539, 1.3522597485901549, 0.9500221754747968, 0.9840203265549855, 1.4480400977338292, 1.0149638892017916, 0.9932006221514178, 1.459056452449548, 1.0248786084459387], rtol = rtol)
                    else
                        @test isapprox(result["objective_value_lyapunov_bound"], 11.258316111574207)
                        @test isapprox(result["primal"].Pg[:], [0.8979870693416395, 1.3432060108971777, 0.9418738115511173, 0.9055318507524539, 1.3522597485901549, 0.9500221754747968, 0.9840203265549855, 1.4480400977338292, 1.0149638892017916, 0.9932006221514178, 1.459056452449548, 1.0248786084459387], rtol = rtol)
                    end
                end

                @testset "ProxALM" begin
                    algparams.mode = :coldstart
                    runinfo = run_proxALM(opfdata, rawdata, modelinfo, algparams)
                    @test isapprox(runinfo.maxviol_c[end], 0.0)
                    @test isapprox(runinfo.x.Pg[:], [0.8979849202781868, 1.3432106598597684, 0.9418713802665164, 0.9055296917633987, 1.3522643856420227, 0.9500197334705451, 0.9840203265723066, 1.4480400977255763, 1.0149638891986126, 0.9932005576574989, 1.4590563750278072, 1.0248785387706203], rtol = rtol)
                    @test isapprox(runinfo.x.ωt[:], [0.0, -0.00012071634376338968, 0.0, -0.00014688369736307614], rtol = rtol)
                    @test isapprox(runinfo.λ.ramping[:], [0.0, 0.0, 0.0, 2.1600116062669975e-6, -7.2856632952422725e-6, 5.0513847355437924e-6], rtol = rtol)
                    @test isapprox(runinfo.λ.ctgs[:], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], rtol = rtol)
                    @test isapprox(runinfo.maxviol_t[end], 2.6878514658212893e-5, rtol = rtol)
                    @test isapprox(runinfo.maxviol_d[end], 7.285428638947877e-6, rtol = rtol)
                    @test runinfo.iter == 5
                end
            end




            K = 1
            algparams.decompCtgs = true
            @testset "$T-period, $K-ctgs, time_link=penalty, ctgs_link=frequency_ctrl, decompCtgs" begin
                modelinfo.num_ctgs = K
                rawdata.ctgs_arr = ctgs_arr[1:modelinfo.num_ctgs]
                opfdata = opf_loaddata(rawdata;
                                       time_horizon_start = 1,
                                       time_horizon_end = T,
                                       load_scale = load_scale,
                                       ramp_scale = ramp_scale)
                
                set_rho!(algparams;
                        ngen = length(opfdata.generators),
                        modelinfo = modelinfo,
                        maxρ_t = maxρ,
                        maxρ_c = maxρ)
            
                @testset "Non-decomposed formulation" begin
                    algparams.mode = :nondecomposed
                    result = solve_fullmodel(opfdata, rawdata, modelinfo, algparams)
                    @test isapprox(result["objective_value_nondecomposed"], 11.258316111574212, rtol = rtol)
                    if solver == "Hiop"
                        @test_broken isapprox(result["primal"].Pg[:], [0.8979870693416382, 1.3432060108971793, 0.9418738115511179, 0.9055318507524525, 1.3522597485901564, 0.9500221754747974, 0.9840203265549852, 1.4480400977338292, 1.014963889201792, 0.9932006221514175, 1.459056452449548, 1.024878608445939], rtol = rtol)
                    else
                        @test isapprox(result["primal"].Pg[:], [0.8979870693416382, 1.3432060108971793, 0.9418738115511179, 0.9055318507524525, 1.3522597485901564, 0.9500221754747974, 0.9840203265549852, 1.4480400977338292, 1.014963889201792, 0.9932006221514175, 1.459056452449548, 1.024878608445939], rtol = rtol)
                    end
                    if solver == "MadNLP" || solver == "MadNLPGPU" || solver == "Hiop"
                        @test_broken isapprox(result["primal"].Zt[:], [0.0, 0.0, 0.0, 2.7857429391709934e-6, 2.353279608425683e-6, 2.023313001658965e-6], rtol = rtol)
                    else
                        @test isapprox(result["primal"].Zt[:], [0.0, 0.0, 0.0, 2.7857429391709934e-6, 2.353279608425683e-6, 2.023313001658965e-6], rtol = rtol)
                    end
                    if solver == "Hiop"
                        @test_broken isapprox(result["primal"].ωt[:], [0.0, -0.00012071650257302939, 0.0, -0.00014688472954291597], rtol = rtol)
                    else
                        @test isapprox(result["primal"].ωt[:], [0.0, -0.00012071650257302939, 0.0, -0.00014688472954291597], rtol = rtol)
                    end
                end
                @testset "Lyapunov bound" begin
                    algparams.mode = :lyapunov_bound
                    result = solve_fullmodel(opfdata, rawdata, modelinfo, algparams)
                    if solver == "Hiop"
                        @test_broken isapprox(result["objective_value_lyapunov_bound"], 11.258316111574207)
                    else
                        @test isapprox(result["objective_value_lyapunov_bound"], 11.258316111574207)
                    end
                    @test isapprox(result["primal"].Pg[:], [0.8979870693416395, 1.3432060108971777, 0.9418738115511173, 0.9055318507524539, 1.3522597485901549, 0.9500221754747968, 0.9840203265549855, 1.4480400977338292, 1.0149638892017916, 0.9932006221514178, 1.459056452449548, 1.0248786084459387], rtol = rtol)
                end
                # MadNLP gets stuck here
                if solver == "Ipopt" || solver == "Hiop"
                @testset "ProxALM" begin
                    algparams.mode = :coldstart
                    runinfo = run_proxALM(opfdata, rawdata, modelinfo, algparams)
                    @test isapprox(runinfo.x.Pg[:], [0.8847566379915904, 1.3458885793645132, 0.9528041613516992, 0.8889877105252308, 1.3510255515317628, 0.9575214693720255, 0.9689518361922401, 1.4504831243374814, 1.0280553147660263, 0.9743778573888332, 1.4570143379662728, 1.0340634613139639], rtol = rtol)
                    @test isapprox(runinfo.x.ωt[:], [0.0, -6.872285646588893e-5, 0.0, -8.763210220007218e-5], rtol = rtol)
                    @test isapprox(runinfo.λ.ramping[:], [0.0, 0.0, 0.0, -0.02526641008086374, -0.03138993075214526, -0.022582337855672596], rtol = rtol)
                    @test isapprox(runinfo.λ.ctgs[:], [0.0, 0.0, 0.0, 0.029837851198013726, 0.0018123712232017784, -0.029641691338447657, 0.0, 0.0, 0.0, 0.03640088659004858, -0.0024938071819119163, -0.030934017357459445], rtol = rtol)
                    if solver == "Hiop"
                        @test_broken isapprox(runinfo.maxviol_c[end], 9.297964943270377e-5)
                        @test_broken isapprox(runinfo.maxviol_t[end], 3.7014553733172306e-6, rtol = rtol)
                    else
                        @test isapprox(runinfo.maxviol_c[end], 9.297964943270377e-5)
                        @test isapprox(runinfo.maxviol_t[end], 3.7014553733172306e-6, rtol = rtol)
                    end
                    @test isapprox(runinfo.maxviol_d[end], 1.2901722926388947e-6, rtol = rtol)
                    if solver == "Ipopt"
                        @test runinfo.iter == 81
                    end
                end
                end # !MadNLP
            end
        end
    end # solver testset
    end
end
MPI.Finalize()
