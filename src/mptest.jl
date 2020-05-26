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

for ramp_scale in [0.001]
    case = ARGS[1]
    T = parse(Int, ARGS[2])
    scen = "../data/mp_demand/"*basename(case)*"_onehour_60"
    rawdata = RawData(case, scen)
    opfdata = opf_loaddata(rawdata; time_horizon_start = 1, time_horizon_end = T, load_scale = 1.0, ramp_scale = ramp_scale)
    opt = Option()
    opt.has_ramping = true
    opt.weight_loadshed = 0
    opt.weight_scencost = 1.0
    opt.weight_freqctrl = 0
    opt.obj_penalty = false
    opt.obj_gencost = true

    T = size(opfdata.Pd, 2)
    #
    # Solve proximal ALM from cold-start
    #
    #=
    begin
        opt.savefile = getDataFilename("coldstart", case, "proxALM", T, options.sc_constr, true, ramp_scale)
        @time x, λ, savedata = runProxALM(opfdata, rawdata, T; options = opt, verbose_level = 2, fullmodel = false, parallel = true)
    end
    =#

    #
    # Proximal ALM in rolling horizon using warm-start
    #
    begin
        maxρ = 0.01
        rolling_horizon_mp_proxALM(case, opfdata, rawdata, T, ramp_scale, maxρ; options = opt, parallel = true)
    end
end
