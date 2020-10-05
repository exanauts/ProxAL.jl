using Distributed

@everywhere using Pkg
@everywhere Pkg.activate(joinpath(dirname(@__FILE__), ".."))
@everywhere Pkg.instantiate()
@everywhere using ProxAL
@everywhere using JuMP, Ipopt
@everywhere using Plots, LaTeXStrings, JLD

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
    algparams.verbose = 3 # level of output: 0 (none), 1 (stdout), 2 (+plots), 3 (+outfiles)
    algparams.optimizer =
                optimizer_with_attributes(Ipopt.Optimizer,
                    "print_level" => Int64(algparams.verbose > 0)*5)
    if algparams.verbose > 1
        outdir = joinpath(dirname(@__FILE__), "./outfiles/")
        if !ispath(outdir)
            mkdir(outdir)
        end
        modelinfo.savefile = outdir * modelinfo.savefile
    end

    ##
    ##  Solve the model
    ##
    if algparams.mode ∈ [:nondecomposed, :lyapunov_bound]
        solve_fullmodel(opfdata, rawdata, modelinfo, algparams)
    elseif algparams.mode == :coldstart
        run_proxALM(opfdata, rawdata, modelinfo, algparams)
        
        if algparams.verbose > 1
            (runinfo.plt === nothing) &&
                (runinfo.plt = initialize_plot())
            zstar, lyapunov_star = NaN, NaN
            if !isempty(runinfo.opt_sol)
                zstar = runinfo.opt_sol["objective_value_nondecomposed"]
            end
            if !isempty(runinfo.lyapunov_sol)
                lyapunov_star = runinfo.lyapunov_sol["objective_value_lyapunov_bound"]
            end
            for iter=1:runinfo.iter
                optimgap = 100.0abs(runinfo.objvalue[iter] - zstar)/abs(zstar)
                lyapunov_gap = 100.0(runinfo.lyapunov[iter] - lyapunov_star)/abs(lyapunov_star)
                push!(runinfo.plt, 1, iter, runinfo.maxviol_t[iter])
                push!(runinfo.plt, 2, iter, runinfo.maxviol_c[iter])
                push!(runinfo.plt, 3, iter, runinfo.maxviol_d[iter])
                push!(runinfo.plt, 4, iter, runinfo.dist_x[iter])
                push!(runinfo.plt, 5, iter, runinfo.dist_λ[iter])
                push!(runinfo.plt, 6, iter, optimgap)
                push!(runinfo.plt, 7, iter, (lyapunov_gap < 0) ? NaN : lyapunov_gap)
                #=
                if algparams.verbose > 2
                    savefile = modelinfo.savefile * ".iter_" * string(runinfo.iter) * ".jld"
                    iter_sol = Dict()
                    iter_sol["x"] = runinfo.x
                    iter_sol["λ"] = runinfo.λ
                    JLD.save(savefile, iter_sol)
                end
                =#
            end
            savefig(runinfo.plt, modelinfo.savefile * ".plot.png")
    end

    return nothing
end

function options_plot(plt)
    fsz = 20
    plot!(plt,
          fontfamily = "Computer-Modern",
          yscale = :log10,
          framestyle = :box,
          ylim = [1e-4, 1e+1],
          xtickfontsize = fsz,
          ytickfontsize = fsz,
          guidefontsize = fsz,
          titlefontsize = fsz,
          legendfontsize = fsz,
          size = (800, 800)
    )
end

function initialize_plot()
    gr()
    label = [L"|p^0_{gt} - p^0_{g,t-1}| - r_g"
             L"|p^k_{gt} - p^0_{gt} - \alpha_g \omega^k_t|"
             L"|\textrm{\sffamily KKT}|"
             L"|x-x^*|"
             L"|\lambda-\lambda^*|"
             L"|c(x)-c(x^*)|/c(x^*)"
             L"|L-L^*|/L^*"]
    any = Array{Any, 1}(undef, length(label))
    any .= Any[[1,1]]
    plt = plot([Inf, Inf], any,
                lab = reshape(label, 1, length(label)),
                lw = 2.5,
                # markersize = 2.5,
                # markershape = :auto,
                xlabel=L"\textrm{\sffamily Iteration}")
    options_plot(plt)
    return plt
end


main()


