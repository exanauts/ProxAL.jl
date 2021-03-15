using Test
using Ipopt
using ExaPF
using JuMP
using ProxAL
using DelimitedFiles, Printf

DATA_DIR = joinpath(dirname(@__FILE__), "..", "data")
case = "case9"
T = 3
ramp_scale = 0.5
load_scale = 1.0
maxρ = 0.1
quad_penalty = 0.1
rtol = 1e-4

# Load data
case_file = joinpath(DATA_DIR, "$(case).m")
load_file = joinpath(DATA_DIR, "mp_demand", "$(case)_oneweek_168")

@testset "Block Model Formulation" begin
    # ctgs_arr = deepcopy(rawdata.ctgs_arr)

    # Model/formulation settings
    modelinfo = ModelParams()
    modelinfo.num_time_periods = T
    modelinfo.load_scale = load_scale
    modelinfo.ramp_scale = ramp_scale
    modelinfo.allow_obj_gencost = true
    modelinfo.allow_constr_infeas = false
    # rho related
    modelinfo.maxρ_t = maxρ
    modelinfo.maxρ_c = maxρ
    # Initialize block OPFs with base OPF solution
    modelinfo.init_opf = false

    # Algorithm settings
    algparams = AlgParams()
    algparams.parallel = false #algparams.parallel = (nprocs() > 1)
    algparams.verbose = 0
    algparams.mode = :coldstart

    modelinfo.case_name = case
    algparams.optimizer = optimizer_with_attributes(
        Ipopt.Optimizer,
        "print_level" => Int64(algparams.verbose > 0)*5,
    )

    K = 0

    # Instantiate primal and dual buffers
    nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams)
    primal = ProxAL.PrimalSolution(nlp)
    dual = ProxAL.DualSolution(nlp)

    modelinfo_local = deepcopy(nlp.modelinfo)
    modelinfo_local.num_time_periods = 1

    @testset "Timestep $t" for t in 1:T
        opfdata_c = ProxAL.opf_loaddata(nlp.rawdata;
                            time_horizon_start = t,
                            time_horizon_end = t,
                            load_scale = load_scale,
                            ramp_scale = ramp_scale)

        local solution, n
        @testset "JuMP Block model" begin
            blockmodel = ProxAL.JuMPBlockModel(1, opfdata_c, nlp.rawdata, modelinfo_local, t, 1, T)
            ProxAL.init!(blockmodel, nlp.algparams)

            ProxAL.set_objective!(blockmodel, nlp.algparams, primal, dual)
            n = JuMP.num_variables(blockmodel.model)
            x0 = zeros(n)
            solution = ProxAL.optimize!(blockmodel, x0, nlp.algparams)
            @test solution.status ∈ ProxAL.MOI_OPTIMAL_STATUSES
        end
        obj_jump = solution.minimum
        pg_jump = solution.pg
        slack_jump = solution.st

        @testset "ExaPF Block model" begin
            # TODO: currently, we need to build directly ExaPF object
            # with rawdata, as ExaPF is dealing only with struct of arrays objects.
            blockmodel = ProxAL.ExaBlockModel(1, opfdata_c, nlp.rawdata, modelinfo_local, t, 1, T)
            ProxAL.init!(blockmodel, nlp.algparams)
            ProxAL.set_objective!(blockmodel, nlp.algparams, primal, dual)

            # Better to set nothing than 0 to avoid convergence issue
            # in the powerflow solver.
            x0 = nothing

            # Set up optimizer
            algparams.gpu_optimizer = optimizer_with_attributes(
                Ipopt.Optimizer,
                "print_level" => 0,
                "limited_memory_max_history" => 50,
                "hessian_approximation" => "limited-memory",
                "derivative_test" => "first-order",
                "tol" => 1e-6,
            )

            solution = ProxAL.optimize!(blockmodel, x0, algparams)
            @test solution.status ∈ ProxAL.MOI_OPTIMAL_STATUSES
        end
        obj_exa = solution.minimum
        pg_exa = solution.pg
        slack_exa = solution.st
        @test obj_jump ≈ obj_exa rtol=1e-3
        @test pg_jump ≈ pg_exa rtol=1e-1
        if t > 1  # slack could be of any value for t == 1
            @test slack_jump ≈ slack_exa rtol=1e-1
        end

        # println()
        # @info("Exa block + AugLag")
        # @testset "ExaPF Block model" begin
        #     blockmodel = ProxAL.ExaBlockModel(1, opfdata_c, rawdata, modelinfo_local, t, 1, T)
        #     ProxAL.init!(blockmodel, algparams)
        #     ProxAL.set_objective!(blockmodel, algparams, primal, dual)
        #     n = ExaPF.n_variables(blockmodel.model)
        #     x0 = zeros(n)

        #     # Set up optimizer
        #     algparams.optimizer = ExaOpt.AugLagSolver(; max_iter=20, ωtol=1e-4, verbose=1)

        #     solution = ProxAL.optimize!(blockmodel, x0, algparams)
        #     println("obj: ", solution.minimum)
        #     println("sol: ", solution.pg)
        # end
    end

    @testset "OPFBlocks" begin
        blocks = ProxAL.OPFBlocks(
            nlp.opfdata, nlp.rawdata;
            modelinfo=nlp.modelinfo,
            algparams=nlp.algparams,
        )

        @test length(blocks.blkModel) == nlp.modelinfo.num_time_periods
    end
end

