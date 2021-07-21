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

include("Evaluators/Evaluators.jl")
include("params.jl")
include("opfdata.jl")
include("opfsolution.jl")
include("blockmodel.jl")
include("blocks.jl")
include("opfmodel.jl")
include("utils.jl")
include("communication.jl")
include("Evaluators/ProxALEvalutor.jl")
include("Evaluators/NonDecomposedModel.jl")

export ModelParams, AlgParams
export ProxALEvaluator, NonDecomposedModel
export optimize!
export JuMPBackend, ExaPFBackend, ExaTronBackend

function update_primal_nlpvars(x::PrimalSolution, opfBlockData::OPFBlocks, blk::Int,
                               modelinfo::ModelParams,
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
    if !algparams.decompCtgs
        @views x.ωt[range_k,t] .= opfBlockData.colValue[from:to,blk]
    end

    from = 1+to
    to = to + size(x.St, 1)
    @views x.St[:,t] .= opfBlockData.colValue[from:to,blk]

    # Zt will be updated in update_primal_penalty
    from = 1+to
    to = to + size(x.Zt, 1)
    # @views x.Zt[:,t] .= opfBlockData.colValue[from:to,blk]

    from = 1+to
    to = to + size(x.Sk, 1)*length(range_k)
    @views x.Sk[:,range_k,t][:] .= opfBlockData.colValue[from:to,blk]

    # Zk will be updated in update_primal_penalty
    # from = 1+to
    # to = to + size(x.Sk, 1)*length(range_k)
    # @views x.Zk[:,range_k,t] .= opfBlockData.colValue[from:to,blk]
    return nothing
end

function update_primal_penalty(x::PrimalSolution, opfdata::OPFData,
                               opfBlockData::OPFBlocks, blk::Int,
                               primal::PrimalSolution,
                               dual::DualSolution,
                               modelinfo::ModelParams,
                               algparams::AlgParams)
    ngen = size(x.Pg, 1)
    block = opfBlockData.blkIndex[blk]
    k = block[1]
    t = block[2]

    if t > 1 && k == 1 && modelinfo.time_link_constr_type == :penalty
        β = [opfdata.generators[g].ramp_agc for g=1:ngen]
        @views x.Zt[:,t] .= (((algparams.ρ_t/32.0)*primal.Zt[:,t]) .- dual.ramping[:,t] .-
                                (algparams.ρ_t*(+x.Pg[:,1,t-1] .- x.Pg[:,1,t] .+ x.St[:,t] .- β))
                            ) ./  max(algparams.zero, (algparams.ρ_t/32.0) + algparams.ρ_t + (modelinfo.obj_scale*algparams.θ_t))
    end
    if k > 1 && algparams.decompCtgs
        if modelinfo.ctgs_link_constr_type == :frequency_ctrl
            x.ωt[k,t] = (( (algparams.ρ_c/32.0)*primal.ωt[k,t]) -
                            sum(opfdata.generators[g].alpha *
                                    (dual.ctgs[g,k,t] + (algparams.ρ_c * (x.Pg[g,1,t] - x.Pg[g,k,t])))
                                for g=1:ngen)
                        ) ./ max(algparams.zero, (algparams.ρ_c/32.0) + (modelinfo.obj_scale*modelinfo.weight_freq_ctrl) +
                                sum(algparams.ρ_c*(opfdata.generators[g].alpha)^2
                                    for g=1:ngen)
                                )
        end
        if modelinfo.ctgs_link_constr_type ∈ [:preventive_penalty, :corrective_penalty]
            if modelinfo.ctgs_link_constr_type == :corrective_penalty
                β = [opfdata.generators[g].scen_agc for g=1:ngen]
            else
                β = zeros(ngen)
                @assert norm(x.Sk) <= algparams.zero
            end
            @views x.Zk[:,k,t] .=   (((algparams.ρ_c/32.0)*primal.Zk[:,k,t]) .- dual.ctgs[:,k,t] .-
                                        (algparams.ρ_c*(+x.Pg[:,1,t] .- x.Pg[:,k,t] .+ x.Sk[:,k,t] .- β))
                                    ) ./  max(algparams.zero, (algparams.ρ_c/32.0) + algparams.ρ_c + (modelinfo.obj_scale*algparams.θ_c))
        end
    end

    return nothing
end

function update_dual_vars(λ::DualSolution, opfdata::OPFData,
                          opfBlockData::OPFBlocks, blk::Int,
                          primal::PrimalSolution,
                          modelinfo::ModelParams,
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
        @assert modelinfo.ctgs_link_constr_type ∈ [:frequency_ctrl, :preventive_penalty, :corrective_penalty]
        link_constr = compute_ctgs_linking_constraints(d, opfdata, modelinfo, k, t)
        λ.ctgs[:,k,t] += algparams.ρ_c*link_constr[:ctgs][:]
        maxviol_c = maximum(abs.(link_constr[:ctgs][:]))
    end

    return maxviol_t, maxviol_c
end

end # module
