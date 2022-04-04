#############################################################################
# Test scripts for ACTIVSg2000 and ACTIVSg10k
#############################################################################

using ProxAL
using DelimitedFiles, Printf
using LinearAlgebra, JuMP, Ipopt
using CatViews
using Logging
using LazyArtifacts

case = "case_ACTIVSg2000"
#case = "case118"

# choose backend
backend = ProxAL.JuMPBackend()
#backend = ProxAL.ExaTronBackend()

# Load case
case_file = joinpath(artifact"ExaData", "ExaData/matpower/case_ACTIVSg2000.m")
load_file = joinpath(@__DIR__, "../data/mp_demand/", "$(case)_Jul_oneweek_168_60min")
#load_file = joinpath(DATA_DIR, "mp_demand", "$(case)_oneweek_168")

# Model/formulation settings
modelinfo = ModelInfo()
modelinfo.num_time_periods = 1
modelinfo.load_scale = 1.0
modelinfo.ramp_scale = 60.0
modelinfo.corr_scale = 0.5
modelinfo.allow_obj_gencost = true
modelinfo.allow_constr_infeas = false
modelinfo.weight_constr_infeas = 1e8
modelinfo.time_link_constr_type = :penalty
modelinfo.ctgs_link_constr_type = :corrective_penalty
modelinfo.allow_line_limits = false
modelinfo.case_name = case
modelinfo.num_ctgs = 0

# Algorithm settings
algparams = AlgParams()
algparams.verbose = 1
algparams.tol = 1e-3
algparams.decompCtgs = false
algparams.iterlim = 100
algparams.tron_rho_pq = 4e3
algparams.tron_rho_pa = 4e4
algparams.tron_outer_iterlim = 30
algparams.tron_inner_iterlim = 1000
algparams.tron_scale = 1e-4
if isa(backend, ProxAL.ExaTronBackend)
    algparams.device = ProxAL.CUDADevice
end
algparams.optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 5) #,  "tol" => 1e-1*algparams.tol)
algparams.mode = :nondecomposed
algparams.init_opf = false

# Set up and solve problem
nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, backend, nothing)
nlp = NonDecomposedModel(case_file, load_file, modelinfo, algparams)
info = ProxAL.optimize!(nlp);

# @show(info.maxviol_t_actual)
