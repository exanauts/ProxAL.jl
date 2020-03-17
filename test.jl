using Printf
using LinearAlgebra
using MathProgBase, JuMP, Ipopt
using LightGraphs, MetaGraphs, Metis
using Plots, Measures, DelimitedFiles
#import Cairo, Fontconfig, GraphPlot, Compose

include("opfdata.jl")
include("partition.jl")
include("params.jl")
include("solution.jl")
include("acopf.jl")
include("kkt_manual.jl")
include("proxALM.jl")
include("aladin.jl")
include("analysis.jl")

case = ARGS[1]
num_partitions = parse(Int, ARGS[2])

for perturbation in [0, 0.01, 0.1, 0.2, 0.5]

	#@printf("starting ALADIN with perturbation = %.2f...\n", perturbation)
	x, λ = runAladin(case, num_partitions, perturbation);

	#@printf("starting Jacobi proximal ALM with perturbation = %.2f...\n", perturbation)
	x, λ = runProxALM(case, num_partitions, perturbation);
end
