using Test
using MPI
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

@testset "Block Model Backends" begin
    # ctgs_arr = deepcopy(rawdata.ctgs_arr)

    # Model/formulation settings
    modelinfo = ModelInfo()
    modelinfo.num_time_periods = T
    modelinfo.load_scale = load_scale
    modelinfo.ramp_scale = ramp_scale
    modelinfo.allow_obj_gencost = true
    modelinfo.allow_constr_infeas = false

    # Algorithm settings
    algparams = AlgParams()
    algparams.verbose = 0
    algparams.mode = :coldstart
    algparams.ρ_t = maxρ
    algparams.ρ_c = maxρ
    algparams.τ = 3.0*maxρ

    modelinfo.case_name = case
    algparams.optimizer = optimizer_with_attributes(
        Ipopt.Optimizer,
        "print_level" => Int64(algparams.verbose > 0)*5,
    )

    K = 0

    # Instantiate primal and dual buffers
    nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, JuMPBackend(), nothing)
    primal = ProxAL.OPFPrimalSolution(nlp)
    dual = ProxAL.OPFDualSolution(nlp)

    modelinfo_local = deepcopy(nlp.modelinfo)
    modelinfo_local.num_time_periods = 1
    blkid = 1

    @testset "Timestep $t" for t in 1:T
        opfdata_c = ProxAL.opf_loaddata(nlp.rawdata;
                            time_horizon_start = t,
                            time_horizon_end = t,
                            load_scale = load_scale,
                            ramp_scale = ramp_scale,
                            corr_scale = modelinfo.corr_scale)

        local solution, n
        @testset "JuMP Block backend" begin
            blockmodel = ProxAL.JuMPBlockBackend(blkid, opfdata_c, nlp.rawdata, algparams, modelinfo_local, t, 1, T)
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

        @testset "ExaPF Block backend" begin
            # TODO: currently, we need to build directly ExaPF object
            # with rawdata, as ExaPF is dealing only with struct of arrays objects.
            blockmodel = ProxAL.ExaBlockBackend(blkid, opfdata_c, nlp.rawdata, algparams, modelinfo_local, t, 1, T)
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
            obj_exa = solution.minimum
            pg_exa = solution.pg
            slack_exa = solution.st
            @test obj_jump ≈ obj_exa
            @test pg_jump ≈ pg_exa rtol=1e-6
            if t > 1  # slack could be of any value for t == 1
                @test slack_jump ≈ slack_exa rtol=1e-5
            end
        end

        @testset "ExaTron BlockModel" begin
            blockmodel = ProxAL.TronBlockBackend(
                blkid, opfdata_c, nlp.rawdata, algparams, modelinfo_local, t, 1, T;
            )
            ProxAL.init!(blockmodel, nlp.algparams)
            ProxAL.set_objective!(blockmodel, nlp.algparams, primal, dual)

            # Test optimization
            x0 = nothing
            solution = ProxAL.optimize!(blockmodel, x0, nlp.algparams)
            @test solution.status ∈ ProxAL.MOI_OPTIMAL_STATUSES
            obj_tron   = solution.minimum
            pg_tron    = solution.pg
            slack_tron = solution.st
            # TODO: implement ProxAL objective in ExaTron
            @test_broken obj_jump ≈ obj_tron
            @test pg_jump ≈ pg_tron rtol=1e-2
            if t > 1  # slack could be of any value for t == 1
                @test slack_jump ≈ slack_tron rtol=1e-2
            end
        end
    end

    @testset "OPFBlocks" begin
        blocks = ProxAL.OPFBlocks(
            nlp.opfdata, nlp.rawdata;
            modelinfo=nlp.modelinfo,
            algparams=nlp.algparams,
            comm=nothing
        )

        @test length(blocks.blkModel) == nlp.modelinfo.num_time_periods
    end
end
