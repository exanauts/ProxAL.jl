using Test
using ProxAL
using DelimitedFiles, Printf
using Distributed
using SharedArrays, LinearAlgebra, JuMP
using CatViews

ENV["GKSwstype"]="nul"

case = "case9"
T = 2
ramp_scale = 0.5
load_scale = 1.0
maxρ = 0.1
quad_penalty = 0.1
rtol = 1e-4

# Load case
case_file = joinpath(dirname(@__FILE__), "../data/$case")
load_file = joinpath(dirname(@__FILE__), "../data/mp_demand/$(case)_oneweek_168")
rawdata = RawData(case_file, load_file)
ctgs_arr = deepcopy(rawdata.ctgs_arr)

# Model/formulation settings
modelinfo = ModelParams()
modelinfo.case_name = case
modelinfo.num_time_periods = T
modelinfo.load_scale = load_scale
modelinfo.ramp_scale = ramp_scale
modelinfo.allow_obj_gencost = true
modelinfo.allow_constr_infeas = false
modelinfo.weight_quadratic_penalty_time = quad_penalty
modelinfo.weight_freq_ctrl = quad_penalty
modelinfo.time_link_constr_type = :penalty
modelinfo.ctgs_link_constr_type = :frequency_ctrl

# Algorithm settings
algparams = AlgParams()
algparams.parallel = false #algparams.parallel = (nprocs() > 1)
algparams.verbose = 0



K = 0
algparams.decompCtgs = false

# @testset "Hiop" begin
#     if haskey(ENV, "JULIA_HIOP_LIBRARY_PATH")
#         include("Hiop.jl")
#     end
# end
@testset "Ipopt" begin
    include("Ipopt.jl")
end

# @testset "MadNLP" begin
#     include("MadNLP.jl")
# end