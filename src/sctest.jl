using Distributed, SharedArrays
using Printf
using LinearAlgebra
using JuMP, Ipopt
using Plots, Measures, DelimitedFiles

include("opfdata.jl")
include("params.jl")
include("mpsolution.jl")
include("scacopf_model.jl")
include("mpproxALM.jl")
include("analysis.jl")

ENV["GKSwstype"]="nul"

for case in [ARGS[1]]
    T = parse(Int, ARGS[2])
    scen = "../data/mp_demand/"*basename(case)*"_onehour_60"
    rawdata = RawData(case, scen)
    if length(rawdata.ctgs_arr) > T
        rawdata.ctgs_arr = rawdata.ctgs_arr[1:T]
    end
    opfdata = opf_loaddata(rawdata)
    opt = Option()
    opt.sc_constr = true
    opt.freq_ctrl = false
    opt.two_block = false
    opt.obj_penalty = false
    opt.weight_scencost = 1/length(rawdata.ctgs_arr)
    opt.weight_loadshed = 2500.0
    opt.weight_freqctrl = 0
    ramp_scale = Float64(opt.freq_ctrl) + 0.1Float64(opt.two_block)
    maxρ = 10.0

    # include base case
    T = length(rawdata.ctgs_arr) + 1

    #
    # Solve proximal ALM from cold-start
    #
    #=
    begin
        opt.savefile = getDataFilename("sc_coldstart", case, "proxALM", T, opt.sc_constr, true, ramp_scale)
        @time x, λ, savedata = runProxALM(opfdata, rawdata, T; options = opt, verbose_level = 2, fullmodel = false, parallel = true)
    end
    =#

    #
    # Proximal ALM in rolling horizon using warm-start
    #
    begin
        rolling_horizon_sc_proxALM(case, opfdata, rawdata, T, ramp_scale, maxρ; options = opt, parallel = true)
    end
end
