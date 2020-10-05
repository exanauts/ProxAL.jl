using Distributed

@everywhere using Pkg
@everywhere Pkg.activate(joinpath(dirname(@__FILE__), ".."))
@everywhere Pkg.instantiate()
@everywhere using ProxAL
@everywhere using JuMP, Ipopt

ENV["GKSwstype"]="nul"

include(joinpath(dirname(@__FILE__), "usage.jl"))
PARSED_ARGS = parse_commandline()

function main()
    case = PARSED_ARGS["case"]
    unit = PARSED_ARGS["time_unit"]
    T = PARSED_ARGS["T"]
    K = PARSED_ARGS["Ctgs"]
    ramp_scale = PARSED_ARGS["ramp_value"]
    load_scale = PARSED_ARGS["load_scale"]
    decompCtgs = PARSED_ARGS["decompCtgs"]
    time_link_constr_type = Symbol(PARSED_ARGS["ramp_constr"])
    ctgs_link_constr_type = Symbol(PARSED_ARGS["Ctgs_constr"])
    maxρ = PARSED_ARGS["auglag_rho"]
    mode = Symbol(PARSED_ARGS["compute_mode"])
    weight_quadratic_penalty_time = PARSED_ARGS["quad_penalty"]
    weight_quadratic_penalty_ctgs = PARSED_ARGS["quad_penalty"]

    if unit == "minute"
        tlim = 60
        load_file = joinpath(dirname(@__FILE__), "../data/mp_demand/$(case)_onehour_60")
    elseif unit == "hour"
        tlim = 168
        load_file = joinpath(dirname(@__FILE__), "../data/mp_demand/$(case)_oneweek_168")
    end
    if T > tlim
        T = tlim
        println("cannot handle T > $tlim when time_unit=$unit. reducing T = $tlim")
    end


    ##
    ## Load the case data
    ##
    case_file = joinpath(dirname(@__FILE__), "../data/$(case)")
    rawdata = RawData(case_file, load_file)
    if K > length(rawdata.ctgs_arr)
        K = length(rawdata.ctgs_arr)
        println(".Ctgs file has only $K ctgs. reducing K = $K")
    end
    rawdata.ctgs_arr = rawdata.ctgs_arr[1:K]
    opfdata = opf_loaddata(rawdata;
                           time_horizon_start = 1,
                           time_horizon_end = T,
                           load_scale = load_scale,
                           ramp_scale = ramp_scale)


    ##
    ## Set up the model parameters
    ##
    modelinfo = ModelParams()
    modelinfo.case_name = case
    modelinfo.savefile = get_unique_name(PARSED_ARGS)
    modelinfo.num_time_periods = T
    modelinfo.num_ctgs = K
    modelinfo.load_scale = load_scale
    modelinfo.ramp_scale = ramp_scale
    modelinfo.allow_obj_gencost = true
    modelinfo.allow_constr_infeas = false
    modelinfo.weight_constr_infeas = 0
    modelinfo.weight_quadratic_penalty_time = weight_quadratic_penalty_time
    modelinfo.weight_quadratic_penalty_ctgs = weight_quadratic_penalty_ctgs
    modelinfo.weight_ctgs = 1.0 # 1/length(rawdata.ctgs_arr)
    modelinfo.weight_freq_ctrl = weight_quadratic_penalty_ctgs
    modelinfo.time_link_constr_type = time_link_constr_type
    modelinfo.ctgs_link_constr_type = ctgs_link_constr_type

    ##
    ## Set up the algorithm parameters
    ##
    algparams = AlgParams()
    algparams.parallel = (nprocs() > 1)
    algparams.decompCtgs = decompCtgs
    set_rho!(algparams;
             ngen = length(opfdata.generators),
             modelinfo = modelinfo,
             maxρ_t = maxρ,
             maxρ_c = maxρ)
    algparams.mode = mode
    algparams.verbose = 1 # level of output: 0 (none), 1 (stdout), 2 (+plots), 3 (+outfiles)
    if algparams.verbose > 1
        outdir = joinpath(dirname(@__FILE__), "./outfiles/")
        if !ispath(outdir)
            mkdir(outdir)
        end
        modelinfo.savefile = outdir * modelinfo.savefile
    end
    algparams.optimizer =
                optimizer_with_attributes(Ipopt.Optimizer,
                    "print_level" => Int64(algparams.verbose > 0)*5)

    ##
    ##  Solve the model
    ##
    if algparams.mode ∈ [:nondecomposed, :lyapunov_bound]
        solve_fullmodel(opfdata, rawdata, modelinfo, algparams)
    elseif algparams.mode == :coldstart
        run_proxALM(opfdata, rawdata, modelinfo, algparams)
    end

    return nothing
end


main()


