module ExaAdmmBackend

using CUDA
using AMDGPU
import MPI
using KernelAbstractions
using ExaAdmm
using ExaTron
const KA = KernelAbstractions

mutable struct ModelProxAL{T,TD,TI,TM} <: ExaAdmm.AbstractOPFModel{T,TD,TI,TM}
    # OPF's part
    info::ExaAdmm.IterationInformation
    solution::ExaAdmm.AbstractSolution{T,TD}

    # Used for multiple dispatch for multi-period case.
    gen_solution::ExaAdmm.AbstractSolution{T,TD}

    n::Int
    nvar::Int

    gen_start::Int
    line_start::Int

    pgmin_curr::TD   # taking ramping into account for rolling horizon
    pgmax_curr::TD   # taking ramping into account for rolling horizon

    grid_data::ExaAdmm.GridData{T,TD,TI,TM}

    membuf::TM      # memory buffer for line kernel
    gen_membuf::TM  # memory buffer for generator kernel

    # Two-Level ADMM
    nvar_u::Int
    nvar_v::Int
    bus_start::Int # this is for varibles of type v.

    # Padded sizes for MPI
    nline_padded::Int
    nvar_u_padded::Int
    nvar_padded::Int

    # ProxAL's part
    s_curr::TD       # slack for ramping
    smin::TD         # slack's lower bound
    smax::TD         # slack's upper bound
    Q_ref::TD
    c_ref::TD

    function ModelProxAL{T,TD,TI,TM}() where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2}}
        return new{T,TD,TI,TM}()
    end
end

function ModelProxAL(
    env::ExaAdmm.AdmmEnv{T,TD,TI,TM}, t::Int, horizon::Int;
    ramp_ratio=0.02,
) where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2}}
    model = ModelProxAL{T,TD,TI,TM}()

    model.grid_data = ExaAdmm.GridData{T,TD,TI,TM}(env)

    model.n = (env.use_linelimit == true) ? 6 : 4
    model.nline_padded = model.grid_data.nline

    # Memory space is padded for the lines as a multiple of # processes.
    if env.use_mpi
        nprocs = MPI.Comm_size(env.comm)
        model.nline_padded = nprocs * div(model.grid_data.nline, nprocs, RoundUp)
    end

    model.nvar = 2*model.grid_data.ngen + 8*model.grid_data.nline
    model.nvar_padded = model.nvar + 8*(model.nline_padded - model.grid_data.nline)
    model.gen_start = 1
    model.line_start = 2*model.grid_data.ngen + 1

    model.pgmin_curr = TD(undef, model.grid_data.ngen)
    model.pgmax_curr = TD(undef, model.grid_data.ngen)
    copyto!(model.pgmin_curr, model.grid_data.pgmin)
    copyto!(model.pgmax_curr, model.grid_data.pgmax)

    model.grid_data.ramp_rate = TD(undef, model.grid_data.ngen)
    model.grid_data.ramp_rate .= ramp_ratio.*model.grid_data.pgmax

    if env.params.obj_scale != 1.0
        model.grid_data.c2 .*= env.params.obj_scale
        model.grid_data.c1 .*= env.params.obj_scale
        model.grid_data.c0 .*= env.params.obj_scale
    end

    # These are only for two-level ADMM.
    model.nvar_u = 2*model.grid_data.ngen + 8*model.grid_data.nline
    model.nvar_u_padded = model.nvar_u + 8*(model.nline_padded - model.grid_data.nline)
    model.nvar_v = 2*model.grid_data.ngen + 4*model.grid_data.nline + 2*model.grid_data.nbus
    model.bus_start = 2*model.grid_data.ngen + 4*model.grid_data.nline + 1

    # Memory space is allocated based on the padded size.
    # XXX
    model.solution =
        ExaAdmm.Solution{T,TD}(model.nvar_padded)
    ExaAdmm.init_solution!(model, model.solution, env.initial_rho_pq, env.initial_rho_va, env.ka_device)
    model.gen_solution = ExaAdmm.EmptyGeneratorSolution{T,TD}()

    model.membuf = TM(undef, (31, model.grid_data.nline))
    fill!(model.membuf, 0.0)
    model.membuf[29,:] .= model.grid_data.rateA

    model.info = ExaAdmm.IterationInformation{ExaAdmm.ComponentInformation}()

    # ProxAL
    ## Slack
    model.s_curr  = TD(undef, model.grid_data.ngen)
    model.smin  = TD(undef, model.grid_data.ngen)
    model.smax  = TD(undef, model.grid_data.ngen)
    # Costs
    model.Q_ref  = TD(undef, 4*model.grid_data.ngen)
    model.c_ref  = TD(undef, 2*model.grid_data.ngen)

    # Don't let any array uninitialized
    for field in [
        :s_curr, :smin, :smax, :Q_ref, :c_ref
    ]
        fill!(getfield(model, field), zero(T))
    end
    return model
end

function ExaAdmm.AdmmEnv(opfdata, rho_va::Float64, rho_pq::Float64; use_gpu=false, ka_device=nothing, options...)
    T = Float64
    if use_gpu
        if !isa(ka_device, Nothing)
            VT = typeof(ExaAdmm.KAArray{Float64}(0, ka_device))
            VI = typeof(ExaAdmm.KAArray{Int}(0, ka_device))
            MT = typeof(ExaAdmm.KAArray{Float64}(0, 0, ka_device))
        else
            VT = CuVector{Float64}
            VI = CuVector{Int}
            MT = CuMatrix{Float64}
        end
    else
        VT = Vector{Float64}
        VI = Vector{Int}
        MT = Matrix{Float64}
    end
    env = ExaAdmm.AdmmEnv{T,VT,VI,MT}(opfdata, "proxal", rho_pq, rho_va; use_gpu=use_gpu, ka_device=ka_device, options...)
    return env
end

include("interface.jl")
include("proxal_admm_cpu.jl")
include("proxal_admm_gpu.jl")

end
