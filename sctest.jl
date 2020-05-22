using Printf
using LinearAlgebra
using MathProgBase, JuMP, Ipopt
using Plots, Measures, DelimitedFiles

include("params.jl")
include("opfdata.jl")
include("mpsolution.jl")
include("scacopf_model.jl")
include("mpproxALM.jl")
include("analysis.jl")
include("kkt_manual.jl")

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
    opt.freq_ctrl = true
    opt.two_block = true
    opt.obj_penalty = false
    opt.weight_scencost = 1/length(rawdata.ctgs_arr)
    opt.weight_loadshed = 2500.0
    opt.weight_freqctrl = 0
    opt.savefile = getDataFilename("", case, "mpproxALM", T, 1.0, true, 0)

    x, λ, savedata = runProxALM_mp(opfdata, rawdata; options = opt)
end
