using Test
using Ipopt
using JuMP
using ProxAL
using DelimitedFiles, Printf

DATA_DIR = joinpath(dirname(@__FILE__), "..", "data")
case = "case9"
T = 2
ramp_scale = 0.5
load_scale = 1.0
maxρ = 0.1
quad_penalty = 0.1
rtol = 1e-4

# Load data
case_file = joinpath(DATA_DIR, "$(case).m")
load_file = joinpath(DATA_DIR, "mp_demand", "$(case)_oneweek_168")
rawdata = RawData(case_file, load_file)
opfdata = opf_loaddata(
    rawdata;
    time_horizon_start = 1,
    time_horizon_end = T,
    load_scale = load_scale,
    ramp_scale = ramp_scale
)

@testset "Model Formulation" begin
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

    modelinfo.case_name = case
    algparams.optimizer = optimizer_with_attributes(Ipopt.Optimizer,
            "print_level" => Int64(algparams.verbose > 0)*5)

    K = 0
    algparams.decompCtgs = false
    @testset "$T-period, $K-ctgs, time_link=penalty" begin
        modelinfo.num_ctgs = K
        rawdata.ctgs_arr = deepcopy(ctgs_arr[1:modelinfo.num_ctgs])

        set_penalty!(algparams;
                 ngen = length(opfdata.generators),
                 modelinfo = modelinfo,
                 maxρ_t = maxρ,
                 maxρ_c = maxρ)

        algparams.mode = :nondecomposed
        result = solve_fullmodel(opfdata, rawdata, modelinfo, algparams)
        @test isapprox(result["objective_value_nondecomposed"], 11.258316111585623, rtol = rtol)
        @test isapprox(result["primal"].Pg[:], [0.8979870694509675, 1.3432060120295906, 0.9418738103137331, 0.9840203268625166, 1.448040098924617, 1.0149638876964715], rtol = rtol)
        @test isapprox(result["primal"].Zt[:], [0.0, 0.0, 0.0, 2.7859277234613066e-6, 2.3533760802049378e-6, 2.0234235436650152e-6], rtol = rtol)
    end
end
