using Printf
using LinearAlgebra
using MathProgBase, JuMP, Ipopt
using Plots, Measures, DelimitedFiles

include("params.jl")
include("mpacopf_data.jl")
include("mpsolution.jl")
include("mpacopf_model.jl")
include("mpproxALM.jl")
include("analysis.jl")
include("kkt_manual.jl")

for ramp_scale in [0.001, 0.002, 0.005]
    case = ARGS[1]
    T = parse(Int, ARGS[2])
    scen = "data/mp_demand/"*basename(case)*"_onehour_60"
    load_scale = 1.0
    baseMVA = 100
    circuit = getcircuit(case, baseMVA, ramp_scale)
    load = getload(scen, load_scale)
    demand = Load(load.pd[:,1:T], load.qd[:,1:T])


    perturbation = 0.1

    #=
    @printf("starting ALADIN with ramp_scale = %.2f...\n", ramp_scale)
	x, λ = runAladin(case, num_partitions, ramp_scale)
    =#

	@printf("starting Jacobi proximal ALM with ramp_scale = %.3f...\n", ramp_scale)
	x, λ, savedata = runProxALM_mp(circuit, demand, ramp_scale, perturbation)
    writedlm(getDataFilename("", case, "mpproxALM", T, perturbation, true, ramp_scale), savedata)
end
