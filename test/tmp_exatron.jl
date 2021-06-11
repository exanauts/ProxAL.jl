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

    @testset "Timestep $t (twolevel=$two_level)" for t in 1:T, two_level in [true]
        opfdata_c = ProxAL.opf_loaddata(nlp.rawdata;
                            time_horizon_start = t,
                            time_horizon_end = t,
                            load_scale = load_scale,
                            ramp_scale = ramp_scale)

        local solution, n
        blockmodel = ProxAL.TronBlockModel(
            1, Array, opfdata_c, nlp.rawdata, modelinfo_local, t, 1, T; iterlim=100, verbose=0,
            use_twolevel=two_level, rho_pq=500.0, rho_va=500.0,
        )
        ProxAL.init!(blockmodel, nlp.algparams)

        # TODO:
        # ProxAL.set_objective!(blockmodel, nlp.algparams, primal, dual)

        # Test optimization
        x0 = nothing
        solution = ProxAL.optimize!(blockmodel, x0, nlp.algparams)
        @test solution.status ∈ ProxAL.MOI_OPTIMAL_STATUSES
        println(solution.pg)

        # Test setters
        @testset "Setters" begin
            env = blockmodel.env
            model = env.model
            ngen = model.gen_mod.ngen
            n = 2 * ngen + 2 * model.nbus
            x0 = zeros(n)
            # Update starting point
            ProxAL.set_start_values!(blockmodel, x0)

            primal = ProxAL.PrimalSolution(opfdata_c, modelinfo)
            dual = ProxAL.DualSolution(opfdata_c, modelinfo)
            ProxAL.update_penalty!(blockmodel, nlp.algparams, primal, dual)
        end
    end
end

