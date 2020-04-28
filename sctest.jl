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

for (idx, case) in enumerate(["data/case9", ARGS[1]])
    T = parse(Int, ARGS[2])
    scen = "data/mp_demand/"*basename(case)*"_onehour_60"
    rawdata = RawData(case, scen)
    if length(rawdata.ctgs_arr) > T
        rawdata.ctgs_arr = rawdata.ctgs_arr[1:T]
    end
    opfdata = opf_loaddata(rawdata)

    if idx > 1
        savefile = getDataFilename("", case, "mpproxALM", T, 0, true, 0)
    else
        savefile = ""
    end
    x, λ, savedata = runProxALM_mp(opfdata, rawdata; sc = true, savefile = savefile)
    if idx > 1
        writedlm(savefile, savedata)
    end
end
