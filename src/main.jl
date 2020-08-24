const current_dir = @__DIR__
using Distributed
@everywhere using Pkg
@everywhere Pkg.activate(current_dir * "/..")
@everywhere Pkg.instantiate()
@everywhere begin
    using SharedArrays
    using Printf
    using LinearAlgebra
    using JuMP, Ipopt
    using Plots, Measures, DelimitedFiles
    using JLD
    using ArgParse

    include("opfdata.jl")
    include("params.jl")
    include("mpsolution.jl")
    include("scacopf_model.jl")
    include("mpproxALM.jl")
    include("analysis.jl")
    include("rolling_horizon.jl")
end

ENV["GKSwstype"]="nul"


function main()
    args = parse_commandline()
    case = args["case"]
    unit = args["time_unit"]
    T = args["T"]
    K = args["Ctgs"]
    ramp_scale = args["ramp_value"]
    load_scale = args["load_scale"]
    ctgs_model = args["Ctgs_model"]
    maxρ = args["rho"]
    mode = args["compute_mode"]
    weight_quadratic_penalty = args["penalty"]


    ##
    ## Consider hard-coding a multiperiod load profile:
    ## multiplier_hourly = [1.0, 1.08199882, 1.11509159, 1.13441867, 1.14755413, 1.15744821, 1.15517453, 1.15771819, 1.15604058, 1.15952428, 1.18587814, 1.22488686, 1.24967452, 1.22919352, 1.16712273, 1.08527516, 0.98558458, 0.89784627, 0.88171420, 0.86553416, 0.8489768, 0.84243794, 0.84521173, 0.90866251]
    ## multiplier_minute = [1.0, 0.99961029, 0.99930348, 0.99907022, 0.99890119, 0.99878704, 0.99871845, 0.99868608, 0.99868059, 0.99869265, 0.99871293, 0.99873216, 0.99874886, 0.99877571, 0.99882691, 0.99891664, 0.9990591, 0.99926107, 0.99949027, 0.99970169, 0.99985026, 0.99989093, 0.99978076, 0.99951609, 0.99912816, 0.99864944, 0.99811239, 0.99754946, 0.99698767, 0.99644131, 0.99592287, 0.99544481, 0.99501961, 0.99465395, 0.99431417, 0.9939494, 0.99350871, 0.99294118, 0.99219837, 0.99130363, 0.99036111, 0.98947927, 0.98876657, 0.98833145, 0.98823171, 0.98837066, 0.98862223, 0.98886034, 0.98895895, 0.98880304, 0.9883816, 0.9877406, 0.98692657, 0.98598606, 0.98496561, 0.98391176, 0.98287107, 0.98189006, 0.98101529, 0.9802933]
    ## Can cyclically repeat the multiplier_hourly profile
    ##
    if unit == "minute"
        tlim = 60
        load_file = current_dir * "/../data/mp_demand/"*case*"_onehour_60"
    elseif unit == "hour"
        tlim = 168
        load_file = current_dir * "/../data/mp_demand/"*case*"_oneweek_168"
    end
    if T > tlim
        T = tlim
        println("cannot handle T > $tlim when time_unit=$unit. reducing T = $tlim")
    end


    ##
    ## Load the case data
    ##
    case_file = current_dir * "../data/" * case
    rawdata = RawData(case_file, load_file)
    if K > length(rawdata.ctgs_arr)
        K = length(rawdata.ctgs_arr)
        println(".Ctgs file has only $K ctgs. reducing K = $K")
    end
    rawdata.ctgs_arr = rawdata.ctgs_arr[1:K]
    opfdata = opf_loaddata(rawdata; time_horizon_start = 1, time_horizon_end = T, load_scale = load_scale, ramp_scale = ramp_scale)


    ##
    ## Set up the model parameters
    ##
    opt = ModelParams()
    opt.num_time_periods = T
    opt.num_ctgs = K
    opt.obj_gencost = true
    opt.allow_constr_infeas = false
    opt.allow_load_shed = false
    opt.add_quadratic_penalty = true
    opt.weight_quadratic_penalty = weight_quadratic_penalty
    opt.weight_scencost = 1.0 # 1/length(rawdata.ctgs_arr)
    opt.weight_loadshed = 0
    opt.weight_freqctrl = 0
    opt.ctgs_link_constr_type = ctgs_model



    ##
    ## Set up the algorithm parameters
    ##
    params = AlgParams()
    params.ρ = maxρ
    params.maxρ = maxρ # consider maxρ = 0.1 for pure sc_constr && freqctrl?
    params.τ = params.jacobi ? 3params.ρ : 0.0
    params.updateρ = (opt.num_time_periods > 1) ? !opt.add_quadratic_penalty : (opt.ctgs_link_constr_type == "corrective")
    params.mode = mode


    ##
    ##  Solve the model
    ##
    if params.mode == "nondecomposed" || params.mode == "lyapunov_bound"
        result = solve_fullmodel(opfdata, rawdata; modelparams = opt, algparams = params)
    elseif params.mode == "coldstart"
        x, λ, savedata =
            runProxALM(opfdata, rawdata;
                       modelparams = opt,
                       algparams = params,
                       verbose_level = 2,
                       fullmodel = false,
                       parallel = true,
                       optfile_x = optfile_x,
                       optfile_L = optfile_L)
    end
end




function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "case"
            help = "Case name [case9, case30, case118, case1354pegase, case2383wp, case9241pegase]"
            required = true
            arg_type = String
            range_tester = x -> x in ["case9", "case30", "case118", "case1354pegase", "case2383wp", "case9241pegase"]
        "--T"
            help = "No. of time periods"
            arg_type = Int
            default = 10
            range_tester = x -> x >= 1
        "--Ctgs"
            help = "No. of line ctgs"
            arg_type = Int
            default = 0
            range_tester = x -> x >= 0
        "--time_unit"
            help = "Select: [hour, minute]"
            default = "minute"
            range_tester = x -> x in ["hour", "minute"]
            metavar = "UNIT"
        "--ramp_value"
            help = "Ramp value: % Pg_max/time_unit"
            default = 0.5
            range_tester = x -> x >= 0
            metavar = "RAMP"
        "--Ctgs_model"
            help = "Select: [preventive, corrective, frequency]"
            arg_type = String
            default = "preventive"
            range_tester = x -> x in ["preventive", "corrective", "frequency"]
            metavar = "MODEL"
        "--load_scale"
            help = "Load multiplier"
            arg_type = Float64
            default = 1.0
            range_tester = x -> x >= 0
            metavar = "SCALE"
        "--penalty"
            help = "Qaudratic penalty parameter"
            arg_type = Float64
            default = 1e3
            range_tester = x -> x >= 0
        "--rho"
            help = "Aug Lag parameter"
            arg_type = Float64
            default = 1.0
            range_tester = x -> x >= 0
        "--compute_mode"
            help = "Choose from: (nondecomposed, coldstart, lyapunov_bound)"
            arg_type = String
            default = "coldstart"
            range_tester = x -> x in ["nondecomposed", "coldstart", "lyapunov_bound"]
            metavar = "MODE"
    end

    return parse_args(ARGS, s)
end




main()
