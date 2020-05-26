#
# Warm-start the proxALM algorithm using the solution computed 
# at the previous time period (with different demand)
#
# function rolling_horizon_mp_proxALM is for multiperiod acopf
# function rolling_horizon_sc_proxALM is for security-constrained acopf
#
#

function rolling_horizon_mp_proxALM(case::String,
                                    opfdata::OPFData,
                                    rawdata::RawData,
                                    T::Int,
                                    ramp_scale::Float64,
                                    maxρ::Float64;
                                    options::Option = Option(),
                                    parallel::Bool = true)
    x = initializePrimalSolution(opfdata, T; options = options)
    λ = initializeDualSolution(opfdata, T; options = options)

    #
    # Option 1: solve the initial full model with ipopt
    #
    x, λ, _, _ = solve_fullmodel(opfdata, rawdata, T; options = opt)
    #=
    if isfile(getDataFilename("optimal_pg", case, "proxALM", T, options.sc_constr, true, ramp_scale))
        x.PG = readdlm(getDataFilename("optimal_pg", case, "proxALM", T, options.sc_constr, true, ramp_scale))
        x.QG = readdlm(getDataFilename("optimal_qg", case, "proxALM", T, options.sc_constr, true, ramp_scale))
        x.VM = readdlm(getDataFilename("optimal_vm", case, "proxALM", T, options.sc_constr, true, ramp_scale))
        x.VA = readdlm(getDataFilename("optimal_va", case, "proxALM", T, options.sc_constr, true, ramp_scale))
        x.SL = readdlm(getDataFilename("optimal_sl", case, "proxALM", T, options.sc_constr, true, ramp_scale))
        λ.λp = readdlm(getDataFilename("optimal_dualp", case, "proxALM", T, options.sc_constr, true, ramp_scale))
        λ.λn = readdlm(getDataFilename("optimal_dualn", case, "proxALM", T, options.sc_constr, true, ramp_scale))
    end
    =#
    Tmax = size(rawdata.pd_arr, 2)
    trange = 2:(Tmax-T+1)


    #
    # Option 2: solve the initial full model with proxALM but increase # of iterations
    #
    # trange = 1:(Tmax-T+1)
    # iter = [(t == 1) ? 100 : 20 for t in trange]


    #
    # Start rolling horizon
    #
    opt = deepcopy(options)
    for t in trange
        opt.savefile = getDataFilename("rolling_horizon_mp_" * string(t), case, "proxALM", T, opt.sc_constr, true, maxρ)
        shiftBackward(x)
        shiftBackward(λ)

        #solve multiperiod problem over horizon t:t+T-1
        opfdata = opf_loaddata(rawdata; time_horizon_start = t, time_horizon_end = t+T-1, ramp_scale = ramp_scale)
        xnew, λnew, _ = runProxALM(opfdata, rawdata, T;
                                   options = opt,
                                   parallel = parallel,
                                   fullmodel = false,
                                   maxρ = maxρ,
                                   initial_x = x,
                                   initial_λ = λ)
        
        x = deepcopy(xnew)
        λ = deepcopy(λnew)
    end
end

function rolling_horizon_sc_proxALM(case::String,
                                    opfdata::OPFData,
                                    rawdata::RawData,
                                    T::Int,
                                    ramp_scale::Float64,
                                    maxρ::Float64;
                                    options::Option = Option(),
                                    parallel::Bool = false)
    x = initializePrimalSolution(opfdata, T; options = options)
    λ = initializeDualSolution(opfdata, T; options = options)

    @assert size(rawdata.pd_arr, 2) > 1

    opt = deepcopy(options)
    for t in 1:2
        opt.savefile = getDataFilename("rolling_horizon_sc_" * string(t), case, "proxALM", T, options.sc_constr, true, maxρ)

        #
        # Fix demand @ time t
        #
        opfdata = opf_loaddata(rawdata; demand_from_rawdata = t)
        xnew, λnew, _ = runProxALM(opfdata, rawdata, T;
                                   options = opt,
                                   verbose_level = verbose_level,
                                   parallel = parallel,
                                   fullmodel = false,
                                   maxρ = maxρ,
                                   initial_x = x,
                                   initial_λ = λ)
        
        x = deepcopy(xnew)
        λ = deepcopy(λnew)
    end

end
