using Distributed
@everywhere using Pkg
@everywhere Pkg.activate("..")
@everywhere Pkg.instantiate()
@everywhere begin
    using SharedArrays
    using Printf
    using LinearAlgebra
    using JuMP, Ipopt
    using Plots, Measures, DelimitedFiles
    using JLD

    include("opfdata.jl")
    include("params.jl")
    include("mpsolution.jl")
    include("scacopf_model.jl")
    include("mpproxALM.jl")
    include("analysis.jl")
    include("rolling_horizon.jl")
end

ENV["GKSwstype"]="nul"

for ramp_scale in [parse(Float64, ARGS[3])]
    case = ARGS[1]
    T = parse(Int, ARGS[2])
    K = parse(Int, ARGS[4])
    load_scale = parse(Float64, ARGS[5])
    weight_quadratic_penalty = parse(Float64, ARGS[6])
    maxρ = parse(Float64, ARGS[7])
    mode = ARGS[8]
    if T > 50
        scen = "../data/mp_demand/"*basename(case)*"_onehour_60"
    else
        scen = "../data/mp_demand/"*basename(case)*"_oneweek_168"
    end
    rawdata = RawData(case, scen)
    rawdata.ctgs_arr = rawdata.ctgs_arr[1:K]
    opfdata = opf_loaddata(rawdata; time_horizon_start = 1, time_horizon_end = T, load_scale = load_scale, ramp_scale = ramp_scale)
    opt = Option()
    opt.has_ramping = true
    opt.quadratic_penalty = true
    opt.weight_quadratic_penalty = weight_quadratic_penalty
    opt.weight_loadshed = 0
    opt.weight_scencost = 1.0
    opt.weight_freqctrl = 0
    opt.obj_penalty = false
    opt.obj_gencost = true

    optfile_x = getDataFilename("./optimalvalues/xstar_", case, "proxALM", T, opt.sc_constr, true, ramp_scale)
    optfile_L = getDataFilename("./optimalvalues/Lstar_rho_" * string(maxρ) * "_", case, "proxALM", T, opt.sc_constr, true, ramp_scale)
    optfile_x = optfile_x[1:end-3] * "jld"
    optfile_L = optfile_L[1:end-3] * "jld"

    T = size(opfdata.Pd, 2)
    ###=
    if mode == "opt" || mode == "Lstar"
        if mode == "opt"
            xstar, λstar, zstar, tfullmodel = solve_fullmodel(opfdata, rawdata, T; options = opt)
            @show(zstar)
            @show(tfullmodel)
            @show(maximum(λstar.λp))
            result = Dict()
            result["xstar"] = xstar
            result["λstar"] = λstar
            result["zstar"] = zstar
            result["tfullmodel"] = tfullmodel
            savefile = optfile_x
        elseif mode == "Lstar"
            Lstar = solve_fullmodel(opfdata, rawdata, T; options = opt, maxρ = maxρ, compute_Lstar = true)
            result = Dict()
            result["Lstar"] = Lstar
            savefile = optfile_L
        end
        JLD.save(savefile, result)
    end
    ##=#
    #
    # Solve proximal ALM from cold-start
    #
    if mode == "coldstart"
        opt.savefile = getDataFilename("./coldstart/rho_" * string(maxρ) * "_", case, "proxALM", T, opt.sc_constr, true, ramp_scale)
        @time x, λ, savedata = runProxALM(opfdata, rawdata, T; options = opt, verbose_level = 2, fullmodel = false, parallel = true, maxρ = maxρ, optfile_x = optfile_x, optfile_L = optfile_L)
    end

    #
    # Proximal ALM in rolling horizon using warm-start
    #
    #=
    begin
        maxρ = 0.01
        rolling_horizon_mp_proxALM(case, opfdata, rawdata, T, ramp_scale, maxρ; options = opt, parallel = true)
    end
    =#
end
