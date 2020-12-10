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
opfdata = opf_loaddata(rawdata;
                       time_horizon_start = 1,
                       time_horizon_end = T,
                       load_scale = load_scale,
                       ramp_scale = ramp_scale)

@testset "Model Formulation" begin
    ctgs_arr = deepcopy(rawdata.ctgs_arr)

    # Model/formulation settings
    modelinfo = ModelParams()
    modelinfo.num_time_periods = 1
    modelinfo.load_scale = load_scale
    modelinfo.ramp_scale = ramp_scale
    modelinfo.allow_obj_gencost = true
    modelinfo.allow_constr_infeas = false
    modelinfo.weight_quadratic_penalty_time = quad_penalty

    # Algorithm settings
    algparams = AlgParams()
    algparams.parallel = false #algparams.parallel = (nprocs() > 1)
    algparams.verbose = 0

    modelinfo.case_name = case
    algparams.optimizer = optimizer_with_attributes(Ipopt.Optimizer,
            "print_level" => Int64(algparams.verbose > 0)*5)

    K = 0

    @testset "Block model" begin
        blockmodel = ProxAL.JuMPBlockModel(1, opfdata, modelinfo, 1, 1)
        ProxAL.init!(blockmodel, algparams)

        # Instantiate primal and dual buffers
        primal = ProxAL.PrimalSolution(opfdata, modelinfo)
        dual = ProxAL.DualSolution(opfdata, modelinfo)

        ProxAL.set_objective!(blockmodel, algparams, primal, dual)
        n = JuMP.num_variables(blockmodel.model)
        x0 = zeros(n)
        ProxAL.optimize!(blockmodel, x0)

    end
end

