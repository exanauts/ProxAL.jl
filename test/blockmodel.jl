using Test
using Ipopt
using ExaPF
using ExaOpt
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
rawdata = RawData(case_file, load_file)
opfdata = opf_loaddata(rawdata;
                       time_horizon_start = 1,
                       time_horizon_end = T,
                       load_scale = load_scale,
                       ramp_scale = ramp_scale)

@testset "Block Model Formulation" begin
    ctgs_arr = deepcopy(rawdata.ctgs_arr)

    # Model/formulation settings
    modelinfo = ModelParams()
    modelinfo.num_time_periods = T
    modelinfo.load_scale = load_scale
    modelinfo.ramp_scale = ramp_scale
    modelinfo.allow_obj_gencost = true
    modelinfo.allow_constr_infeas = false

    # Algorithm settings
    algparams = AlgParams()
    algparams.parallel = false #algparams.parallel = (nprocs() > 1)
    algparams.verbose = 1

    modelinfo.case_name = case
    algparams.optimizer = optimizer_with_attributes(Ipopt.Optimizer,
            "print_level" => Int64(algparams.verbose > 0)*5)

    K = 0

    # Instantiate primal and dual buffers
    primal = ProxAL.PrimalSolution(opfdata, modelinfo)
    dual = ProxAL.DualSolution(opfdata, modelinfo)

    set_rho!(algparams;
             ngen = length(opfdata.generators),
             modelinfo = modelinfo,
             maxρ_t = maxρ,
             maxρ_c = maxρ)

    modelinfo_local = deepcopy(modelinfo)
    modelinfo_local.num_time_periods = 1

    t = 3
    opfdata_c = opf_loaddata(rawdata;
                        time_horizon_start = t,
                        time_horizon_end = t,
                        load_scale = load_scale,
                        ramp_scale = ramp_scale)

    @info("JuMP block + Ipopt")
    @testset "JuMP Block model" begin
        blockmodel = ProxAL.JuMPBlockModel(1, opfdata_c, rawdata, modelinfo_local, t, 1, T)
        ProxAL.init!(blockmodel, algparams)

        ProxAL.set_objective!(blockmodel, algparams, primal, dual)
        n = JuMP.num_variables(blockmodel.model)
        x0 = zeros(n)
        solution = ProxAL.optimize!(blockmodel, x0, algparams)
        println("obj: ", solution.minimum)
        println("sol: ", solution.pg)
    end

    println()
    @info("Exa block + Ipopt")
    @testset "ExaPF Block model" begin
        # TODO: currently, we need to build directly ExaPF object
        # with rawdata, as ExaPF is dealing only with struct of arrays objects.
        blockmodel = ProxAL.ExaBlockModel(1, opfdata_c, rawdata, modelinfo_local, t, 1, T)
        ProxAL.init!(blockmodel, algparams)
        ProxAL.set_objective!(blockmodel, algparams, primal, dual)

        n = ExaPF.n_variables(blockmodel.model)
        x0 = zeros(n)

        # Set up optimizer
        algparams.optimizer = optimizer_with_attributes(
            Ipopt.Optimizer,
            "print_level" => 5,
            "limited_memory_max_history" => 50,
            "hessian_approximation" => "limited-memory",
            "derivative_test" => "first-order",
            "tol" => 1e-6,
        )

        solution = ProxAL.optimize!(blockmodel, x0, algparams)
        println("obj: ", solution.minimum)
        println("sol: ", solution.pg)
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

    println()

    @testset "OPFBlocks" begin
        blocks = ProxAL.OPFBlocks(
            opfdata, rawdata;
            modelinfo=modelinfo,
            algparams=algparams,
        )

        @test length(blocks.blkModel) == modelinfo.num_time_periods
    end
end

