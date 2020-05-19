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

for ramp_scale in [0.001, 0.002, 0.005]
    case = ARGS[1]
    T = parse(Int, ARGS[2])
    scen = "data/mp_demand/"*basename(case)*"_onehour_60"
    rawdata = RawData(case, scen)
    opfdata = opf_loaddata(rawdata; time_horizon = T, load_scale = 1.0, ramp_scale = ramp_scale)
    perturbation = 0.1
    opt = Option()
    opt.has_ramping = true
    opt.weight_loadshed = 0
    opt.weight_scencost = 1.0
    opt.weight_freqctrl = 0
    opt.obj_penalty = false
    opt.obj_gencost = true
    opt.savefile = getDataFilename("", case, "mpproxALM", T, perturbation, true, ramp_scale)



    #=
    @printf("starting ALADIN with ramp_scale = %.2f...\n", ramp_scale)
	x, λ = runAladin(case, num_partitions, ramp_scale)
    =#

	@printf("starting Jacobi proximal ALM with ramp_scale = %.3f...\n", ramp_scale)
    x, λ, savedata = runProxALM_mp(opfdata, rawdata, perturbation; options = opt)
end
