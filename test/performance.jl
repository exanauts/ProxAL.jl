
using Test
using MPI
using Ipopt
using ExaPF
using ExaAdmm
using JuMP
using ProxAL
using DelimitedFiles, Printf
using LazyArtifacts
using BenchmarkTools

const DATA_DIR = joinpath(artifact"ExaData", "ExaData")
case = "case9"
T = 3
ramp_scale = 0.5
load_scale = 1.0
maxρ = 0.1
quad_penalty = 0.1
rtol = 1e-4

# Load data
case_file = joinpath(DATA_DIR, "$(case).m")
load_file = joinpath(DATA_DIR, "mp_demand", "$(case)_oneweek_168")

modelinfo = ModelInfo()
modelinfo.num_time_periods = T
modelinfo.load_scale = load_scale
modelinfo.ramp_scale = ramp_scale
modelinfo.allow_obj_gencost = true
modelinfo.allow_constr_infeas = false

# Algorithm settings
algparams = AlgParams()
algparams.verbose = 0
algparams.mode = :coldstart
algparams.ρ_t = maxρ
algparams.ρ_c = maxρ
algparams.τ = 3.0*maxρ
algparams.tron_outer_iterlim = 30

modelinfo.case_name = case
algparams.optimizer = optimizer_with_attributes(
    Ipopt.Optimizer,
    "print_level" => Int64(algparams.verbose > 0)*5,
)

K = 0

# Instantiate primal and dual buffers
nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, JuMPBackend(), nothing)
primal = ProxAL.OPFPrimalSolution(nlp)
dual = ProxAL.OPFDualSolution(nlp)

modelinfo_local = deepcopy(nlp.modelinfo)
modelinfo_local.num_time_periods = 1
blkid = 1
modelinfo_local.obj_scale = 1e0

# @testset "Timestep $t" for t in 1:T
t= 1
opfdata_c = ProxAL.opf_loaddata(nlp.rawdata;
                    time_horizon_start = t,
                    time_horizon_end = t,
                    load_scale = load_scale,
                    ramp_scale = ramp_scale,
                    corr_scale = modelinfo.corr_scale)

# JuMP
blockmodel = ProxAL.JuMPBlockBackend(blkid, opfdata_c, nlp.rawdata, algparams, modelinfo_local, t, 1, T)
ProxAL.init!(blockmodel, nlp.algparams)
ProxAL.set_objective!(blockmodel, nlp.algparams, primal, dual)
n = JuMP.num_variables(blockmodel.model)
x0 = zeros(n)
solution = ProxAL.optimize!(blockmodel, x0, nlp.algparams)
solution.status ∈ ProxAL.MOI_OPTIMAL_STATUSES
obj_jump = solution.minimum
pg_jump = solution.pg
slack_jump = solution.st
@btime begin
    blockmodel = $blockmodel
    nlp = $nlp
    primal = $primal
    dual = $dual
    x0 = $x0
    obj_jump = $obj_jump
    pg_jump = $pg_jump
    slack_jump  = $slack_jump
    ProxAL.init!(blockmodel, nlp.algparams)
    ProxAL.set_objective!(blockmodel, nlp.algparams, primal, dual)
    solution = ProxAL.optimize!(blockmodel, x0, nlp.algparams)
    solution.status ∈ ProxAL.MOI_OPTIMAL_STATUSES
    @test obj_jump == solution.minimum
    @test pg_jump == solution.pg
    @test slack_jump == solution.st
end


# ExaAdmm CPU
blockmodel = ProxAL.AdmmBlockBackend(
    blkid, opfdata_c, nlp.rawdata, algparams, modelinfo_local, t, 1, T;
)
ProxAL.init!(blockmodel, nlp.algparams)
ProxAL.set_objective!(blockmodel, nlp.algparams, primal, dual)
x0 = nothing
solution = ProxAL.optimize!(blockmodel, x0, nlp.algparams)
@test solution.status ∈ ProxAL.MOI_OPTIMAL_STATUSES
@test obj_jump ≈ solution.minimum rtol=1e-4
@test pg_jump ≈ solution.pg rtol=1e-3
@btime begin
    alparams = $algparams
    modelinfo_local = $modelinfo_local
    t = $t
    T = $T
    opfdata_c = $opfdata_c
    blkid = $blkid
    nlp = $nlp
    primal = $primal
    dual = $dual
    x0 = $x0
    obj_jump = $obj_jump
    pg_jump = $pg_jump
    slack_jump  = $slack_jump
    blockmodel = ProxAL.AdmmBlockBackend(
        blkid, opfdata_c, nlp.rawdata, algparams, modelinfo_local, t, 1, T;
    )
    ProxAL.init!(blockmodel, nlp.algparams)
    ProxAL.set_objective!(blockmodel, nlp.algparams, primal, dual)
    x0 = nothing
    solution = ProxAL.optimize!(blockmodel, x0, nlp.algparams)
    @test solution.status ∈ ProxAL.MOI_OPTIMAL_STATUSES
    @test obj_jump ≈ solution.minimum rtol=1e-4
    @test pg_jump ≈ solution.pg rtol=1e-3
end

# ExaAdmm GPU CUDA
algparams.device = ProxAL.GPU
blockmodel = ProxAL.AdmmBlockBackend(
    blkid, opfdata_c, nlp.rawdata, algparams, modelinfo_local, t, 1, T;
)
ProxAL.init!(blockmodel, nlp.algparams)
ProxAL.set_objective!(blockmodel, nlp.algparams, primal, dual)
x0 = nothing
solution = ProxAL.optimize!(blockmodel, x0, nlp.algparams)
@test solution.status ∈ ProxAL.MOI_OPTIMAL_STATUSES
@test obj_jump ≈ solution.minimum rtol=1e-4
@test pg_jump ≈ solution.pg rtol=1e-3
# blockmodel = ProxAL.AdmmBlockBackend(
#     blkid, opfdata_c, nlp.rawdata, algparams, modelinfo_local, t, 1, T;
# )
@btime begin
    blockmodel = $blockmodel
    alparams = $algparams
    modelinfo_local = $modelinfo_local
    t = $t
    T = $T
    opfdata_c = $opfdata_c
    blkid = $blkid
    nlp = $nlp
    primal = $primal
    dual = $dual
    x0 = $x0
    obj_jump = $obj_jump
    pg_jump = $pg_jump
    slack_jump  = $slack_jump
    blockmodel = ProxAL.AdmmBlockBackend(
        blkid, opfdata_c, nlp.rawdata, algparams, modelinfo_local, t, 1, T;
    )
    ProxAL.init!(blockmodel, nlp.algparams)
    ProxAL.set_objective!(blockmodel, nlp.algparams, primal, dual)
    x0 = nothing
    solution = ProxAL.optimize!(blockmodel, x0, nlp.algparams)
    @test solution.status ∈ ProxAL.MOI_OPTIMAL_STATUSES
    @test obj_jump ≈ solution.minimum rtol=1e-4
    @test pg_jump ≈ solution.pg rtol=1e-3
end

# ExaAdmm GPU KA
algparams.device = ProxAL.GPU
using CUDA
using AMDGPU
if CUDA.has_cuda_gpu()
    using CUDAKernels
    function ProxAL.ExaAdmm.KAArray{T}(n::Int, device::CUDADevice) where {T}
        return CuArray{T}(undef, n)
    end
    function ProxAL.ExaAdmm.KAArray{T}(n1::Int, n2::Int, device::CUDADevice) where {T}
        return CuArray{T}(undef, n1, n2)
    end
    gpu_device = CUDADevice()
elseif AMDGPU.has_rocm_gpu()
    using ROCKernels
    # Set for crusher login node to avoid other users
    AMDGPU.default_device!(AMDGPU.devices()[2])
    function ProxAL.ExaAdmm.KAArray{T}(n::Int, device::ROCDevice) where {T}
        return ROCArray{T}(undef, n)
    end
    function ProxAL.ExaAdmm.KAArray{T}(n1::Int, n2::Int, device::ROCDevice) where {T}
        return ROCArray{T}(undef, n1, n2)
    end
    gpu_device = ROCDevice()
end
using CUDAKernels
algparams.ka_device = gpu_device
blockmodel = ProxAL.AdmmBlockBackend(
    blkid, opfdata_c, nlp.rawdata, algparams, modelinfo_local, t, 1, T;
)
ProxAL.init!(blockmodel, nlp.algparams)
ProxAL.set_objective!(blockmodel, nlp.algparams, primal, dual)
x0 = nothing
solution = ProxAL.optimize!(blockmodel, x0, nlp.algparams)
@test solution.status ∈ ProxAL.MOI_OPTIMAL_STATUSES
@test obj_jump ≈ solution.minimum rtol=1e-4
@test pg_jump ≈ solution.pg rtol=1e-3
# blockmodel = ProxAL.AdmmBlockBackend(
#     blkid, opfdata_c, nlp.rawdata, algparams, modelinfo_local, t, 1, T;
# )
@btime begin
    blockmodel = $blockmodel
    alparams = $algparams
    modelinfo_local = $modelinfo_local
    t = $t
    T = $T
    opfdata_c = $opfdata_c
    blkid = $blkid
    nlp = $nlp
    primal = $primal
    dual = $dual
    x0 = $x0
    obj_jump = $obj_jump
    pg_jump = $pg_jump
    slack_jump  = $slack_jump
    blockmodel = ProxAL.AdmmBlockBackend(
        blkid, opfdata_c, nlp.rawdata, algparams, modelinfo_local, t, 1, T;
    )
    ProxAL.init!(blockmodel, nlp.algparams)
    ProxAL.set_objective!(blockmodel, nlp.algparams, primal, dual)
    x0 = nothing
    solution = ProxAL.optimize!(blockmodel, x0, nlp.algparams)
    @test solution.status ∈ ProxAL.MOI_OPTIMAL_STATUSES
    @test obj_jump ≈ solution.minimum rtol=1e-4
    @test pg_jump ≈ solution.pg rtol=1e-3
end
