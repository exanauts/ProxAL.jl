#
# proximal ALM implementation
#
module ProxAL

using Ipopt, JuMP
using Printf, CatViews
using ExaPF
using ExaAdmm
using KernelAbstractions
using LinearAlgebra
using SparseArrays
using MPI
using HDF5
const KA = KernelAbstractions

abstract type AbstractPrimalSolution end
abstract type AbstractDualSolution end
abstract type AbstractBackend end
abstract type AbstractBlocks end
mutable struct ProxALProblem
    opfBlockData::AbstractBlocks

    #---- iterate information ----
    x::AbstractPrimalSolution
    λ::AbstractDualSolution
    xprev::AbstractPrimalSolution
    λprev::AbstractDualSolution
    objvalue::Vector{Float64}
    lyapunov::Vector{Float64}
    maxviol_t::Vector{Float64}
    maxviol_c::Vector{Float64}
    maxviol_d::Vector{Float64}
    maxviol_t_actual::Vector{Float64}
    maxviol_c_actual::Vector{Float64}
    minviol::Float64
    nlp_soltime::Vector{Float64}
    wall_time_elapsed_actual::Float64
    wall_time_elapsed_ideal::Float64
    iter::Int64

    #---- other/static information ----
    blkLinIndices::LinearIndices{2, Tuple{UnitRange{Int64}, UnitRange{Int64}}}
    blkLocalIndices::Union{Vector{Int64}, Nothing}
    blkLinkedIndices::Union{Vector{Int64}, Nothing}

    # write output
    output::Bool
end
include("communication.jl")
include("Evaluators/Evaluators.jl")
include("ExaAdmmBackend/ExaAdmmBackend.jl")
include("params.jl")
include("OPF/opfdata.jl")
include("backends.jl")
include("blocks.jl")
include("LocalStorage.jl")
include("OPF/opfsolution.jl")
include("OPF/opfmodel.jl")
include("Evaluators/ProxALEvalutor.jl")
include("Evaluators/NonDecomposedModel.jl")

export ModelInfo, AlgParams
export ProxALEvaluator, NonDecomposedModel
export optimize!
export JuMPBackend, ExaPFBackend, AdmmBackend
export write

function update_primal_penalty(
    x::AbstractPrimalSolution,
    opfdata::OPFData,
    opfBlockData::AbstractBlocks,
    blk::Int,
    primal::AbstractPrimalSolution,
    dual::AbstractDualSolution,
    modelinfo::ModelInfo,
    algparams::AlgParams
)
    ngen = size(x.Pg, 1)
    block = opfBlockData.blkIndex[blk]
    k = block[1]
    t = block[2]

    if t > 1 && k == 1 && modelinfo.time_link_constr_type == :penalty
        β = [opfdata.generators[g].ramp_agc for g=1:ngen]
        x.Zt[:,t] = (((algparams.ρ_t/32.0)*primal.Zt[:,t]) .- dual.ramping[:,t] .-
                                (algparams.ρ_t*(x.Pg[:,1,t-1] .- x.Pg[:,1,t] .+ x.St[:,t] .- β))
                            ) ./  max(algparams.zero, (algparams.ρ_t/32.0) + algparams.ρ_t + (modelinfo.obj_scale*algparams.θ_t))
    end
    if k > 1 && algparams.decompCtgs
        if modelinfo.ctgs_link_constr_type ∈ [:frequency_penalty, :preventive_penalty, :corrective_penalty]
            if modelinfo.ctgs_link_constr_type == :corrective_penalty
                β = [opfdata.generators[g].scen_agc for g=1:ngen]
            elseif modelinfo.ctgs_link_constr_type == :frequency_penalty
                β = [-opfdata.generators[g].alpha*x.ωt[k,t] for g=1:ngen]
                @assert norm(x.Sk[:,k,t]) <= algparams.zero
            else
                β = zeros(ngen)
                @assert norm(x.Sk[:,k,t]) <= algparams.zero
            end
            x.Zk[:,k,t] = (((algparams.ρ_c/32.0)*primal.Zk[:,k,t]) .- dual.ctgs[:,k,t] .-
                                        (algparams.ρ_c*(x.Pg[:,1,t] .- x.Pg[:,k,t] .+ x.Sk[:,k,t] .- β))
                                    ) ./  max(algparams.zero, (algparams.ρ_c/32.0) + algparams.ρ_c + (modelinfo.obj_scale*algparams.θ_c))
        end
    end

    return nothing
end

function update_dual_vars(
    λ::AbstractDualSolution,
    opfdata::OPFData,
    opfBlockData::AbstractBlocks,
    blk::Int,
    primal::AbstractPrimalSolution,
    modelinfo::ModelInfo,
    algparams::AlgParams
)
    block = opfBlockData.blkIndex[blk]
    k = block[1]
    t = block[2]

    maxviol_t, maxviol_c = 0.0, 0.0

    if t > 1 || (k > 1 && algparams.decompCtgs)
        d = Dict(:Pg => primal.Pg,
                 :ωt => primal.ωt,
                 :St => primal.St,
                 :Zt => primal.Zt,
                 :Sk => primal.Sk,
                 :Zk => primal.Zk)
    end

    if t > 1 && k == 1
        @assert modelinfo.time_link_constr_type == :penalty
        link_constr = compute_time_linking_constraints(d, opfdata, modelinfo, t)
        λ.ramping[:,t] += algparams.ρ_t*link_constr[:ramping][:]
        maxviol_t = maximum(abs.(link_constr[:ramping][:]))
    end
    if k > 1 && algparams.decompCtgs
        @assert modelinfo.ctgs_link_constr_type ∈ [:frequency_penalty, :preventive_penalty, :corrective_penalty]
        link_constr = compute_ctgs_linking_constraints(d, opfdata, modelinfo, k, t)
        λ.ctgs[:,k,t] += algparams.ρ_c*link_constr[:ctgs][:]
        maxviol_c = maximum(abs.(link_constr[:ctgs][:]))
    end

    return maxviol_t, maxviol_c
end


function ProxALProblem(
    opfdata::OPFData,
    rawdata::RawData,
    modelinfo::ModelInfo,
    algparams::AlgParams,
    backend::AbstractBackend,
    comm::Union{MPI.Comm,Nothing};
    output = false
)
    # initial values
    backend = if isa(backend, JuMPBackend)
        JuMPBlockBackend
    elseif isa(backend, ExaPFBackend)
        ExaBlockBackend
    elseif isa(backend, AdmmBackend)
        AdmmBlockBackend
    end
    # NLP blocks
    blocks = OPFBlocks(
        opfdata, rawdata;
        modelinfo=modelinfo, algparams=algparams,
        backend=backend, comm=comm
    )
    # Linearize Cartesian indices
    blkLinIndices = LinearIndices(blocks.blkIndex)
    # Only do fully distributed if contingencies are decomposed (and MPI enabled)
    # Get the local indices this process has to work on
    blkLocalIndices = Vector{Int64}()
    for blk in blkLinIndices
        if is_my_work(blk, comm)
            push!(blkLocalIndices, blk)
        end
    end
    # Get the list of neighbors this process has to communicate with
    blkLinkedIndices = Vector{Int64}()
    for blk in blkLocalIndices
        block = blocks.blkIndex[blk]
        k = block[1]
        t = block[2]
        for blkn in blkLinIndices
            blockn = blocks.blkIndex[blkn]
            kn = blockn[1]
            tn = blockn[2]
            if is_comm_pattern(t, tn, k, kn, CommPatternTK()) && !is_my_work(blkn, comm)
                push!(blkLinkedIndices, blkn)
            end
        end
    end
    if !algparams.decompCtgs == true || !isa(comm, MPI.Comm)
        x = OPFPrimalSolution(opfdata, modelinfo, blocks, blkLocalIndices, blkLinkedIndices, Array)
        λ = OPFDualSolution(opfdata, modelinfo, blocks, blkLocalIndices, blkLinkedIndices, Array)
    else
        x = OPFPrimalSolution(opfdata, modelinfo, blocks, blkLocalIndices, blkLinkedIndices, LocalStorage)
        λ = OPFDualSolution(opfdata, modelinfo, blocks, blkLocalIndices, blkLinkedIndices, LocalStorage)
    end
    for blk in blkLocalIndices
        if algparams.mode ∉ [:nondecomposed, :lyapunov_bound]
            init!(blocks.blkModel[blk], algparams)
        end
    end

    iter = 0
    objvalue = []
    lyapunov = []
    maxviol_t = []
    maxviol_c = []
    maxviol_d = []
    maxviol_t_actual = []
    maxviol_c_actual = []
    nlp_soltime = Vector{Float64}(undef, blocks.blkCount)
    wall_time_elapsed_actual = 0.0
    wall_time_elapsed_ideal = 0.0
    xprev = deepcopy(x)
    λprev = deepcopy(λ)

    return ProxALProblem(
        blocks,
        x,
        λ,
        xprev,
        λprev,
        objvalue,
        lyapunov,
        maxviol_t,
        maxviol_c,
        maxviol_d,
        maxviol_t_actual,
        maxviol_c_actual,
        Inf,
        nlp_soltime,
        wall_time_elapsed_actual,
        wall_time_elapsed_ideal,
        iter,
        blkLinIndices,
        blkLocalIndices,
        blkLinkedIndices,
        output
    )
end

function runinfo_update(
    runinfo::ProxALProblem,
    opfdata::OPFData,
    opfBlockData::AbstractBlocks,
    modelinfo::ModelInfo,
    algparams::AlgParams,
    comm::Union{MPI.Comm,Nothing}
)
    iter = runinfo.iter
    obj = 0.0
    for blk in runinfo.blkLocalIndices
        obj += compute_objective_function(runinfo.x, opfdata, opfBlockData, blk, modelinfo, algparams)
    end
    obj = comm_sum(obj, comm)
    push!(runinfo.objvalue, obj)

    iter = runinfo.iter
    lyapunov = 0.0
    for blk in runinfo.blkLocalIndices
        lyapunov += compute_lyapunov_function(runinfo.x, runinfo.λ, opfdata, opfBlockData, blk, runinfo.xprev, modelinfo, algparams)
    end
    lyapunov = comm_sum(lyapunov, comm)
    push!(runinfo.lyapunov, lyapunov)

    maxviol_t_actual = 0.0
    for blk in runinfo.blkLocalIndices
        lmaxviol_t_actual = compute_true_ramp_error(runinfo.x, opfdata, opfBlockData, blk, modelinfo)
        maxviol_t_actual = max(maxviol_t_actual, lmaxviol_t_actual)
    end
    maxviol_t_actual = comm_max(maxviol_t_actual, comm)
    push!(runinfo.maxviol_t_actual, maxviol_t_actual)

    maxviol_c_actual = 0.0
    for blk in runinfo.blkLocalIndices
        lmaxviol_c_actual = compute_true_ctgs_error(runinfo.x, opfdata, opfBlockData, blk, modelinfo)
        maxviol_c_actual = max(maxviol_c_actual, lmaxviol_c_actual)
    end
    maxviol_c_actual = comm_max(maxviol_c_actual, comm)
    push!(runinfo.maxviol_c_actual, maxviol_c_actual)

    maxviol_d = 0.0
    for blk in runinfo.blkLocalIndices
        lmaxviol_d = compute_dual_error(runinfo.x, runinfo.xprev, runinfo.λ, runinfo.λprev, opfdata, opfBlockData, blk, modelinfo, algparams)
        maxviol_d = max(maxviol_d, lmaxviol_d)
    end
    maxviol_d = comm_max(maxviol_d, comm)
    push!(runinfo.maxviol_d, maxviol_d)
    if algparams.verbose > 0 && comm_rank(comm) == 0
        if iter == 1
            @printf("---------------------------------------------------------------------------------------------------------------------------\n");
            @printf("iter ramp_err   ramp_err   ctgs_err   ctgs_err   dual_error lyapunov_f    rho_t    rho_c  theta_t  theta_c      tau minviol\n");
            @printf("     (penalty)  (actual)   (penalty)  (actual)\n");
            @printf("---------------------------------------------------------------------------------------------------------------------------\n");
        end
        @printf("%4d ", iter-1);
        @printf("%10.4e ", runinfo.maxviol_t[iter])
        @printf("%10.4e ", runinfo.maxviol_t_actual[iter])
        @printf("%10.4e ", runinfo.maxviol_c[iter])
        @printf("%10.4e ", runinfo.maxviol_c_actual[iter])
        @printf("%10.4e ", runinfo.maxviol_d[iter])
        @printf("%10.4e ", runinfo.lyapunov[iter])
        @printf("%8.2e ", algparams.ρ_t)
        @printf("%8.2e ", algparams.ρ_c)
        @printf("%8.2e ", algparams.θ_t)
        @printf("%8.2e ", algparams.θ_c)
        @printf("%8.2e ", algparams.τ)
        @printf("%10.4e ", runinfo.minviol)
        @printf("\n")
    end
end

end # module
