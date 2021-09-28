#
# proximal ALM implementation
#
module ProxAL

using Ipopt, JuMP
using Printf, CatViews
using ExaPF
using ExaTron
using LinearAlgebra
using SparseArrays
using MPI

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
    dist_x::Vector{Float64}
    dist_λ::Vector{Float64}
    nlp_opt_sol::Array{Float64,2}
    nlp_soltime::Vector{Float64}
    wall_time_elapsed_actual::Float64
    wall_time_elapsed_ideal::Float64
    iter

    #---- other/static information ----
    opt_sol::Dict
    lyapunov_sol::Dict
    initial_solve::Bool
    par_order
    plt
end

include("Evaluators/Evaluators.jl")
include("params.jl")
include("OPF/opfdata.jl")
include("OPF/opfsolution.jl")
include("backends.jl")
include("blocks.jl")
include("OPF/opfmodel.jl")
include("communication.jl")
include("Evaluators/ProxALEvalutor.jl")
include("Evaluators/NonDecomposedModel.jl")

export ModelInfo, AlgParams
export ProxALEvaluator, NonDecomposedModel
export optimize!
export JuMPBackend, ExaPFBackend, ExaTronBackend

function update_primal_nlpvars(x::AbstractPrimalSolution, opfBlockData::AbstractBlocks, blk::Int,
                               modelinfo::ModelInfo,
                               algparams::AlgParams)
    block = opfBlockData.blkIndex[blk]
    k = block[1]
    t = block[2]
    range_k = algparams.decompCtgs ? (k:k) : (1:(modelinfo.num_ctgs + 1))

    from = 1
    to = size(x.Pg,1)*length(range_k)
    @views x.Pg[:,range_k,t][:] .= opfBlockData.colValue[from:to,blk]

    from = 1+to
    to = to + size(x.Qg, 1)*length(range_k)
    @views x.Qg[:,range_k,t][:] .= opfBlockData.colValue[from:to,blk]

    from = 1+to
    to = to + size(x.Vm, 1)*length(range_k)
    @views x.Vm[:,range_k,t][:] .= opfBlockData.colValue[from:to,blk]

    from = 1+to
    to = to + size(x.Va, 1)*length(range_k)
    @views x.Va[:,range_k,t][:] .= opfBlockData.colValue[from:to,blk]

    from = 1+to
    to = to + length(range_k)
    @views x.ωt[range_k,t] .= opfBlockData.colValue[from:to,blk]

    from = 1+to
    to = to + size(x.St, 1)
    if !algparams.decompCtgs || k == 1
        @views x.St[:,t] .= opfBlockData.colValue[from:to,blk]
    end

    # Zt will be updated in update_primal_penalty
    from = 1+to
    to = to + size(x.Zt, 1)

    from = 1+to
    to = to + size(x.Sk, 1)*length(range_k)
    if !algparams.decompCtgs || k > 1
        @views x.Sk[:,range_k,t][:] .= opfBlockData.colValue[from:to,blk]
    end

    from = 1+to
    to = to + size(x.Zk, 1)*length(range_k)
    if !algparams.decompCtgs
        @views x.Zk[:,range_k,t][:] .= opfBlockData.colValue[from:to,blk]
    end

    return nothing
end

function update_primal_penalty(x::AbstractPrimalSolution, opfdata::OPFData,
                               opfBlockData::AbstractBlocks, blk::Int,
                               primal::AbstractPrimalSolution,
                               dual::AbstractDualSolution,
                               modelinfo::ModelInfo,
                               algparams::AlgParams)
    ngen = size(x.Pg, 1)
    block = opfBlockData.blkIndex[blk]
    k = block[1]
    t = block[2]

    if t > 1 && k == 1 && modelinfo.time_link_constr_type == :penalty
        β = [opfdata.generators[g].ramp_agc for g=1:ngen]
        @views x.Zt[:,t] .= (((algparams.ρ_t/32.0)*primal.Zt[:,t]) .- dual.ramping[:,t] .-
                                (algparams.ρ_t*(x.Pg[:,1,t-1] .- x.Pg[:,1,t] .+ x.St[:,t] .- β))
                            ) ./  max(algparams.zero, (algparams.ρ_t/32.0) + algparams.ρ_t + (modelinfo.obj_scale*algparams.θ_t))
    end
    if k > 1 && algparams.decompCtgs
        if modelinfo.ctgs_link_constr_type ∈ [:frequency_penalty, :preventive_penalty, :corrective_penalty]
            if modelinfo.ctgs_link_constr_type == :corrective_penalty
                β = [opfdata.generators[g].scen_agc for g=1:ngen]
            elseif modelinfo.ctgs_link_constr_type == :frequency_penalty
                β = [-opfdata.generators[g].alpha*x.ωt[k,t] for g=1:ngen]
                @assert norm(x.Sk) <= algparams.zero
            else
                β = zeros(ngen)
                @assert norm(x.Sk) <= algparams.zero
            end
            @views x.Zk[:,k,t] .=   (((algparams.ρ_c/32.0)*primal.Zk[:,k,t]) .- dual.ctgs[:,k,t] .-
                                        (algparams.ρ_c*(x.Pg[:,1,t] .- x.Pg[:,k,t] .+ x.Sk[:,k,t] .- β))
                                    ) ./  max(algparams.zero, (algparams.ρ_c/32.0) + algparams.ρ_c + (modelinfo.obj_scale*algparams.θ_c))
        end
    end

    return nothing
end

function update_dual_vars(λ::AbstractDualSolution, opfdata::OPFData,
                          opfBlockData::AbstractBlocks, blk::Int,
                          primal::AbstractPrimalSolution,
                          modelinfo::ModelInfo,
                          algparams::AlgParams)
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
    opfdata::OPFData, rawdata::RawData,
    modelinfo::ModelInfo,
    algparams::AlgParams,
    backend::AbstractBackend,
    comm::Union{MPI.Comm,Nothing},
    opt_sol = Dict(),
    lyapunov_sol = Dict(),
    initial_primal = nothing,
    initial_dual = nothing
)
    if !isempty(opt_sol)
        (algparams.verbose > 0) &&
            @printf("Optimal objective value = %.2f\n", opt_sol["objective_value_nondecomposed"])
    end
    if !isempty(lyapunov_sol)
        (algparams.verbose > 0) &&
            @printf("Lyapunov lower bound = %.2f\n", lyapunov_sol["objective_value_lyapunov_bound"])
    end

    # initial values
    initial_solve = initial_primal === nothing
    x = (initial_primal === nothing) ?
            OPFPrimalSolution(opfdata, modelinfo) :
            deepcopy(initial_primal)
    λ = (initial_dual === nothing) ?
            OPFDualSolution(opfdata, modelinfo) :
            deepcopy(initial_dual)
    backend = if isa(backend, JuMPBackend)
        JuMPBlockBackend
    elseif isa(backend, ExaPFBackend)
        ExaBlockBackend
    elseif isa(backend, ExaTronBackend)
        TronBlockBackend
    end
    # NLP blocks
    blocks = OPFBlocks(
        opfdata, rawdata;
        modelinfo=modelinfo, algparams=algparams,
        backend=backend, comm
    )
    blkLinIndex = LinearIndices(blocks.blkIndex)
    for blk in blkLinIndex
        if is_my_work(blk, comm)
            # model = blocks.blkModel[blk]
            # init!(model, algparams)
            blocks.colValue[:,blk] .= get_block_view(x, blocks.blkIndex[blk], modelinfo, algparams)
        end
    end

    par_order = []
    if algparams.jacobi
        par_order = blkLinIndex
    elseif algparams.decompCtgs
        par_order = @view blkLinIndex[2:end,:]
    end
    plt = nothing
    iter = 0
    objvalue = []
    lyapunov = []
    maxviol_t = []
    maxviol_c = []
    maxviol_d = []
    maxviol_t_actual = []
    maxviol_c_actual = []
    dist_x = []
    dist_λ = []
    nlp_opt_sol = Array{Float64, 2}(undef, blocks.colCount, blocks.blkCount)
    nlp_opt_sol .= blocks.colValue
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
        dist_x,
        dist_λ,
        nlp_opt_sol,
        nlp_soltime,
        wall_time_elapsed_actual,
        wall_time_elapsed_ideal,
        iter,
        opt_sol,
        lyapunov_sol,
        initial_solve,
        par_order,
        plt
    )
end

function runinfo_update(
    runinfo::ProxALProblem, opfdata::OPFData,
    opfBlockData::AbstractBlocks,
    modelinfo::ModelInfo,
    algparams::AlgParams,
    comm::Union{MPI.Comm,Nothing}
)
    iter = runinfo.iter
    obj = 0.0
    for blk in runinfo.par_order
        if is_my_work(blk, comm)
            obj += compute_objective_function(runinfo.x, opfdata, opfBlockData, blk, modelinfo, algparams)
        end
    end
    obj = comm_sum(obj, comm)
    push!(runinfo.objvalue, obj)

    iter = runinfo.iter
    lyapunov = 0.0
    for blk in runinfo.par_order
        if is_my_work(blk, comm)
            lyapunov += compute_lyapunov_function(runinfo.x, runinfo.λ, opfdata, opfBlockData, blk, runinfo.xprev, modelinfo, algparams)
        end
    end
    lyapunov = comm_sum(lyapunov, comm)
    push!(runinfo.lyapunov, lyapunov)

    maxviol_t_actual = 0.0
    for blk in runinfo.par_order
        if is_my_work(blk, comm)
            lmaxviol_t_actual = compute_true_ramp_error(runinfo.x, opfdata, opfBlockData, blk, modelinfo)
            maxviol_t_actual = max(maxviol_t_actual, lmaxviol_t_actual)
        end
    end
    maxviol_t_actual = comm_max(maxviol_t_actual, comm)
    push!(runinfo.maxviol_t_actual, maxviol_t_actual)

    maxviol_c_actual = 0.0
    for blk in runinfo.par_order
        if is_my_work(blk, comm)
            lmaxviol_c_actual = compute_true_ctgs_error(runinfo.x, opfdata, opfBlockData, blk, modelinfo)
            maxviol_c_actual = max(maxviol_c_actual, lmaxviol_c_actual)
        end
    end
    maxviol_c_actual = comm_max(maxviol_c_actual, comm)
    push!(runinfo.maxviol_c_actual, maxviol_c_actual)

    # maxviol_d = compute_dual_error(runinfo.x, runinfo.xprev, runinfo.λ, runinfo.λprev, opfdata, modelinfo, algparams)
    # maxviol_d = comm_max(maxviol_d, comm)
    maxviol_d = 0.0
    for blk in runinfo.par_order
        if is_my_work(blk, comm)
            lmaxviol_d = compute_dual_error(runinfo.x, runinfo.xprev, runinfo.λ, runinfo.λprev, opfdata, opfBlockData, blk, modelinfo, algparams)
            maxviol_d = max(maxviol_d, lmaxviol_d)
        end
    end
    maxviol_d = comm_max(maxviol_d, comm)
    push!(runinfo.maxviol_d, maxviol_d)

    push!(runinfo.dist_x, NaN)
    push!(runinfo.dist_λ, NaN)
    #=
    optimgap = NaN
    lyapunov_gap = NaN
    if !isempty(runinfo.opt_sol)
        xstar = runinfo.opt_sol["primal"]
        zstar = runinfo.opt_sol["objective_value_nondecomposed"]
        runinfo.dist_x[iter] = dist(runinfo.x, xstar, modelinfo, algparams)
        optimgap = 100.0 * abs(runinfo.objvalue[iter] - zstar) / abs(zstar)
    end
    if !isempty(runinfo.lyapunov_sol)
        lyapunov_star = runinfo.lyapunov_sol["objective_value_lyapunov_bound"]
        lyapunov_gap = 100.0 * (runinfo.lyapunov[end] - lyapunov_star) / abs(lyapunov_star)
    end

    if algparams.verbose > 0 && comm_rank(comm) == 0
        @printf("iter %3d: ramp_err = %.3e, ctgs_err = %.3e, dual_err = %.3e, |x-x*| = %.3f, gap = %.2f%%, lyapgap = %.2f%%\n",
                    iter,
                    runinfo.maxviol_t[iter],
                    runinfo.maxviol_c[iter],
                    runinfo.maxviol_d[iter],
                    runinfo.dist_x[iter],
                    optimgap,
                    lyapunov_gap)
    end
    =#
    if algparams.verbose > 0 && comm_rank(comm) == 0
        if iter == 1
            @printf("--------------------------------------------------------------------------------------------------------------------\n");
            @printf("iter ramp_err   ramp_err   ctgs_err   ctgs_err   dual_error lyapunov_f    rho_t    rho_c  theta_t  theta_c      tau \n");
            @printf("     (penalty)  (actual)   (penalty)  (actual)\n");
            @printf("--------------------------------------------------------------------------------------------------------------------\n");
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
        @printf("\n")
    end
end

end # module
