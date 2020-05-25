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

for ramp_scale in [0.001, 0.002, 0.005]
    case = ARGS[1]
    T = parse(Int, ARGS[2])
    scen = "data/mp_demand/"*basename(case)*"_onehour_60"
    rawdata = RawData(case, scen)
    opfdata = opf_loaddata(rawdata; time_horizon = T, load_scale = 1.0, ramp_scale = ramp_scale)
    opt = Option()
    opt.has_ramping = true
    opt.weight_loadshed = 0
    opt.weight_scencost = 1.0
    opt.weight_freqctrl = 0
    opt.obj_penalty = false
    opt.obj_gencost = true
    opt.savefile = getDataFilename("", case, "proxALM", T, false, true, ramp_scale)

    T = size(opfdata.Pd, 2)
    @time x, λ, savedata = runProxALM(opfdata, rawdata, T; options = opt)
end
