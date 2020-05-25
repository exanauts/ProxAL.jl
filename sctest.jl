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

for (idx, case) in enumerate([ARGS[1]])
    T = parse(Int, ARGS[2])
    scen = "data/mp_demand/"*basename(case)*"_onehour_60"
    rawdata = RawData(case, scen)
    if length(rawdata.ctgs_arr) > T
        rawdata.ctgs_arr = rawdata.ctgs_arr[1:T]
    end
    opfdata = opf_loaddata(rawdata)
    opt = Option()
    opt.sc_constr = true
    opt.freq_ctrl = false
    opt.two_block = true
    opt.obj_penalty = false
    opt.weight_scencost = 0#1/length(rawdata.ctgs_arr)
    opt.weight_loadshed = 2500.0
    opt.weight_freqctrl = 0
    opt.savefile = getDataFilename("", case, "proxALM", T, sc_constr, true, 0)

    # include base case
    T = length(rawdata.ctgs_arr) + 1
    @time x, λ, savedata = runProxALM(opfdata, rawdata, T; options = opt)
end
