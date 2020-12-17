#
# proximal ALM implementation
#
module ProxAL

using JuMP
using Printf, CatViews
using ExaPF
using LinearAlgebra
using SparseArrays
using MPI

include("params.jl")
include("opfdata.jl")
include("opfsolution.jl")
include("opfmodel.jl")
include("blockmodel.jl")
include("full_model.jl")
include("blocks.jl")
include("proxALMutil.jl")

export RawData, ModelParams, AlgParams
export opf_loaddata, solve_fullmodel, run_proxALM, set_rho!

"""
    run_proxALM(opfdata::OPFData,
                rawdata::RawData,
                modelinfo::ModelParams,
                algparams::AlgParams,
                comm::MPI.Comm = MPI.COMM_WORLD)

Use ProxAL to solve the multi-period ACOPF instance
specified in `opfdata` and `rawdata` with model parameters
`modelinfo` and algorithm parameters `algparams`, and 
an MPI communicator `comm`.
"""
function run_proxALM(opfdata::OPFData, rawdata::RawData,
                     modelinfo::ModelParams,
                     algparams::AlgParams,
                     comm::MPI.Comm = MPI.COMM_WORLD)
    runinfo = ProxALMData(opfdata, rawdata, modelinfo, algparams, comm, false)
    runinfo.initial_solve &&
        (algparams_copy = deepcopy(algparams))
    opfBlockData = runinfo.opfBlockData
    nlp_opt_sol = runinfo.nlp_opt_sol
    nlp_soltime = runinfo.nlp_soltime
    ngen = length(opfdata.generators)
    nbus = length(opfdata.buses)
    T = modelinfo.num_time_periods
    K = modelinfo.num_ctgs + 1 # base case counted separately
    x = runinfo.x
    λ = runinfo.λ

    function transfer!(blk, opt_sol, solution)
        # Pg
        fr = 1 ; to = ngen * K
        opt_sol[fr:to, blk] .= solution.pg[:]
        # Qg
        fr = to + 1 ; to += ngen * K
        opt_sol[fr:to, blk] .= solution.qg[:]
        # vm
        fr = to +1 ; to = fr + nbus * K - 1
        opt_sol[fr:to, blk] .= solution.vm[:]
        # va
        fr = to + 1 ; to = fr + nbus * K - 1
        opt_sol[fr:to, blk] .= solution.va[:]
        # wt
        fr = to +1  ; to = fr + K -1
        opt_sol[fr:to, blk] .= solution.ωt[:]
        # St
        fr = to +1  ; to = fr + ngen - 1
        opt_sol[fr:to, blk] .= solution.st[:]
    end
    #------------------------------------------------------------------------------------
    function blocknlp_copy(blk, x_ref, λ_ref, alg_ref)
        model = opfBlockData.blkModel[blk]
        # Update objective
        set_objective!(model, alg_ref, x_ref, λ_ref)
        x0 = @view opfBlockData.colValue[:, blk]
        solution = optimize!(model, x0, alg_ref)
        transfer!(blk, nlp_opt_sol, solution)
    end
    #------------------------------------------------------------------------------------
    function blocknlp_recreate(blk, x_ref, λ_ref, alg_ref)
        model = opfBlockData.blkModel[blk]
        init!(model, alg_ref)
        # Update objective
        set_objective!(model, alg_ref, x_ref, λ_ref)
        x0 = @view opfBlockData.colValue[:, blk]
        solution = optimize!(model, x0, alg_ref)
        transfer!(blk, nlp_opt_sol, solution)
    end
    #------------------------------------------------------------------------------------
    function primal_update()
        runinfo.wall_time_elapsed_actual += @elapsed begin
            # Primal update except penalty vars
            for blk in runinfo.ser_order
                nlp_soltime[blk] = @elapsed blocknlp_copy(blk, x, λ, algparams)
                # nlp_soltime[blk] = @elapsed blocknlp_recreate(blk; x_ref = x, λ_ref = λ, alg_ref = algparams)
                opfBlockData.colValue[:,blk] .= nlp_opt_sol[:,blk]
                if !algparams.jacobi
                    update_primal_nlpvars(x, opfBlockData, blk, modelinfo, algparams)
                end
            end
            nlp_opt_sol .= 0.0
            nlp_soltime .= 0.0
            for blk in runinfo.par_order
                if blk % MPI.Comm_size(comm) == MPI.Comm_rank(comm)
                    # nlp_soltime[blk] = @elapsed blocknlp_copy(blk; x_ref = x, λ_ref = λ, alg_ref = algparams)
                    nlp_soltime[blk] = @elapsed blocknlp_recreate(blk, x, λ, algparams)
                end
            end

            # Every worker sends his contribution
            MPI.Allreduce!(nlp_opt_sol, MPI.SUM, comm)
            MPI.Allreduce!(nlp_soltime, MPI.SUM, comm)

            for blk in runinfo.par_order
                opfBlockData.colValue[:,blk] .= nlp_opt_sol[:,blk]
                update_primal_nlpvars(x, opfBlockData, blk, modelinfo, algparams)
            end
            if algparams.jacobi
                for blk in runinfo.ser_order
                    update_primal_nlpvars(x, opfBlockData, blk, modelinfo, algparams)
                end
            end

            # Primal update of penalty vars
            elapsed_t = @elapsed update_primal_penalty(x, opfdata, x, λ, modelinfo, algparams)
        end
        runinfo.wall_time_elapsed_ideal += isempty(runinfo.ser_order) ? 0.0 : sum(nlp_soltime[blk] for blk in runinfo.ser_order)
        runinfo.wall_time_elapsed_ideal += isempty(runinfo.par_order) ? 0.0 : maximum([nlp_soltime[blk] for blk in runinfo.par_order])
        runinfo.wall_time_elapsed_ideal += elapsed_t
    end
    #------------------------------------------------------------------------------------
    function dual_update()
        elapsed_t = @elapsed begin
            maxviol_t, maxviol_c = update_dual_vars(λ, opfdata, x, modelinfo, algparams)
            push!(runinfo.maxviol_t, maxviol_t)
            push!(runinfo.maxviol_c, maxviol_c)
        end
        runinfo.wall_time_elapsed_actual += elapsed_t
        runinfo.wall_time_elapsed_ideal += elapsed_t
    end
    #------------------------------------------------------------------------------------
    function proximal_update()
        elapsed_t = @elapsed begin
            if algparams.updateτ
                maxρ = algparams.decompCtgs ? max(maxρ_t, maxρ_c) : maxρ_t
                # lyapunov = compute_lyapunov_function(runinfo.x, runinfo.λ, opfdata; xref = runinfo.xprev, modelinfo = modelinfo, algparams = algparams)
                # delta_lyapunov = runinfo.lyapunovprev - lyapunov
                # if delta_lyapunov <= 0.0 && algparams.τ < 3.0maxρ
                if runinfo.iter%10 == 0 && algparams.τ < 3.0maxρ
                    algparams.τ += maxρ
                end
            end
        end
        runinfo.wall_time_elapsed_actual += elapsed_t
        runinfo.wall_time_elapsed_ideal += elapsed_t
    end
    #------------------------------------------------------------------------------------


    for runinfo.iter=1:algparams.iterlim

        if runinfo.initial_solve
            if runinfo.iter == 1
                set_rho!(algparams;
                         maxρ_t = 0.0,
                         maxρ_c = 0.0,
                         ngen = length(opfdata.generators),
                         modelinfo = modelinfo)
                algparams.updateρ_t = false
                algparams.updateρ_c = false
            elseif runinfo.iter == 2
                algparams = deepcopy(algparams_copy)
            end
        end

        # use this to compute the KKT error at the end of the loop
        runinfo.xprev = deepcopy(x)
        runinfo.λprev = deepcopy(λ)

        # Primal update
        primal_update()

        # Dual update
        dual_update()

        # Prox update
        proximal_update()

        # Update counters and write output
        update_runinfo(runinfo, opfdata, modelinfo, algparams)

        # Check convergence
        if max(runinfo.maxviol_t[end], runinfo.maxviol_c[end], runinfo.maxviol_d[end]) <= algparams.tol
            break
        end
    end

    return runinfo
end

function update_primal_nlpvars(x::PrimalSolution, opfBlockData::OPFBlocks, blk::Int,
                               modelinfo::ModelParams,
                               algparams::AlgParams)
    solution = get_block_view(x, opfBlockData.blkIndex[blk],
                              modelinfo,
                              algparams)
    solution .= opfBlockData.colValue[:,blk]

    return nothing
end

function update_primal_penalty(x::PrimalSolution, opfdata::OPFData,
                               primal::PrimalSolution,
                               dual::DualSolution,
                               modelinfo::ModelParams,
                               algparams::AlgParams)
    (ngen, K, T) = size(x.Pg)

    if modelinfo.time_link_constr_type == :penalty
        β = [opfdata.generators[g].ramp_agc for g=1:ngen]
        @views for t=2:T
            x.Zt[:,t] .= ((algparams.τ*primal.Zt[:,t]) .- dual.ramping[:,t] .-
                            (algparams.ρ_t[:,t].*(+x.Pg[:,1,t-1] .- x.Pg[:,1,t] .+ x.St[:,t] .- β))
                        ) ./  max.(algparams.zero, algparams.τ .+ algparams.ρ_t[:,t] .+ (modelinfo.obj_scale*modelinfo.weight_quadratic_penalty_time))
        end
    end
    if K > 1 && algparams.decompCtgs
        if modelinfo.ctgs_link_constr_type == :frequency_ctrl
            @views for k=2:K
                x.ωt[k,:] .= (( algparams.τ*primal.ωt[k,:]) .-
                                sum(opfdata.generators[g].alpha *
                                        (dual.ctgs[g,k,:] .+ (algparams.ρ_c[g,k,:] .* (x.Pg[g,1,:] .- x.Pg[g,k,:])))
                                    for g=1:ngen)
                            ) ./ max.(algparams.zero, algparams.τ .+ (modelinfo.obj_scale*modelinfo.weight_freq_ctrl) .+
                                    sum(algparams.ρ_c[g,k,:]*(opfdata.generators[g].alpha)^2
                                        for g=1:ngen)
                                )
            end
        end
        if modelinfo.ctgs_link_constr_type ∈ [:preventive_penalty, :corrective_penalty]
            β = zeros(ngen,T)
            if modelinfo.ctgs_link_constr_type == :corrective_penalty
                for g=1:ngen
                    β[g,:] .= opfdata.generators[g].scen_agc
                end
            else
                @assert norm(x.Sk) <= algparams.zero
            end
            @views for k=2:K
                x.Zk[:,k,:] .= ((algparams.τ*primal.Zk[:,k,:]) .- dual.ctgs[:,k,:] .-
                                (algparams.ρ_c[:,k,:].*(+x.Pg[:,1,:] .- x.Pg[:,k,:] .+ x.Sk[:,k,:] .- β))
                            ) ./  max.(algparams.zero, algparams.τ .+ algparams.ρ_c[:,k,:] .+ (modelinfo.obj_scale*modelinfo.weight_quadratic_penalty_ctgs))
            end
        end
    end

    return nothing
end

function update_dual_vars(λ::DualSolution, opfdata::OPFData,
                          primal::PrimalSolution,
                          modelinfo::ModelParams,
                          algparams::AlgParams)
    (ngen, K, T) = size(primal.Pg)

    maxviol_t, maxviol_c = 0.0, 0.0

    if T > 1 || (K > 1 && algparams.decompCtgs)
        d = Dict(:Pg => primal.Pg,
                 :ωt => primal.ωt,
                 :St => primal.St,
                 :Zt => primal.Zt,
                 :Sk => primal.Sk,
                 :Zk => primal.Zk)
    end

    if T > 1
        link_constr = compute_time_linking_constraints(d, opfdata, modelinfo)
        viol_t = (modelinfo.time_link_constr_type == :inequality) ?
                    max.(link_constr[:ramping_p], link_constr[:ramping_n], 0.0) :
                    abs.(link_constr[:ramping])
        if algparams.updateρ_t
            increaseρ_t = (algparams.ρ_t .< algparams.maxρ_t) .& (viol_t .> algparams.ρ_t_tol)
            subρ_t = @view algparams.ρ_t[increaseρ_t]
            subρ_t .= min.(subρ_t .+ 0.1algparams.maxρ_t, algparams.maxρ_t)
            subρ_t_tol = @view algparams.ρ_t_tol[(.!increaseρ_t) .& (viol_t .<= algparams.ρ_t_tol)]
            subρ_t_tol .= max.(subρ_t_tol./1.2, algparams.zero)
        end

        if modelinfo.time_link_constr_type == :inequality
            λ.ramping_p .= max.(λ.ramping_p .+ (algparams.θ*algparams.ρ_t.*link_constr[:ramping_p]), 0)
            λ.ramping_n .= max.(λ.ramping_n .+ (algparams.θ*algparams.ρ_t.*link_constr[:ramping_n]), 0)
        else
            λ.ramping += algparams.θ*algparams.ρ_t.*link_constr[:ramping]
        end

        maxviol_t = maximum(viol_t)
    end
    if K > 1 && algparams.decompCtgs
        link_constr = compute_ctgs_linking_constraints(d, opfdata, modelinfo)
        viol_c = (modelinfo.ctgs_link_constr_type == :corrective_inequality) ?
                    max.(link_constr[:ctgs_p], link_constr[:ctgs_n], 0.0) :
                    abs.(link_constr[:ctgs])
        if algparams.updateρ_c
            increaseρ_c = (algparams.ρ_c .< algparams.maxρ_c) .& (viol_c .> algparams.ρ_c_tol)
            subρ_c = @view algparams.ρ_c[increaseρ_c]
            subρ_c .= min.(subρ_c .+ 0.1algparams.maxρ_c, algparams.maxρ_c)
            subρ_c_tol = @view algparams.ρ_c_tol[(.!increaseρ_c) .& (viol_c .<= algparams.ρ_c_tol)]
            subρ_c_tol .= max.(subρ_c_tol./1.2, algparams.zero)
        end

        if modelinfo.time_link_constr_type == :corrective_inequality
            λ.ctgs_p .= max.(λ.ctgs_p .+ (algparams.θ*algparams.ρ_c.*link_constr[:ctgs_p]), 0)
            λ.ctgs_n .= max.(λ.ctgs_n .+ (algparams.θ*algparams.ρ_c.*link_constr[:ctgs_n]), 0)
        else
            λ.ctgs += algparams.θ*algparams.ρ_c.*link_constr[:ctgs]
        end

        maxviol_c = maximum(viol_c)
    end

    return maxviol_t, maxviol_c
end

end # module
