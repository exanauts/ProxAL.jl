struct ProxALEvaluator <: AbstractNLPEvaluator
    problem::ProxALProblem
    modelinfo::ModelInfo
    algparams::AlgParams
    opfdata::OPFData
    rawdata::RawData
    space::AbstractBackend
    comm::Union{MPI.Comm,Nothing}
end

"""
    ProxALEvaluator(
        case_file::String,
        load_file::String,
        modelinfo::ModelInfo,
        algparams::AlgParams,
        space::AbstractBackend = JuMPBackend(),
        comm::Union{MPI.Comm,Nothing} = MPI.COMM_WORLD
    )

Instantiate multi-period ACOPF
specified in `case_file` with loads in `load_file` with model parameters
`modelinfo`, algorithm parameters `algparams`, modeling backend `space`, and
a MPI communicator `comm`.

"""
function ProxALEvaluator(
    case_file::String,
    load_file::String,
    modelinfo::ModelInfo,
    algparams::AlgParams,
    space::AbstractBackend=JuMPBackend(),
    comm::Union{MPI.Comm,Nothing} = MPI.COMM_WORLD;
    output::Bool = false
)
    rawdata = RawData(case_file, load_file)
    return ProxALEvaluator(rawdata, modelinfo, algparams, space, comm; output=output)
end

"""
    ProxALEvaluator(
        rawdata::RawData,
        modelinfo::ModelInfo,
        algparams::AlgParams,
        space::AbstractBackend = JuMPBackend(),
        comm::Union{MPI.Comm,Nothing} = MPI.COMM_WORLD
    )

Instantiate multi-period ACOPF using `rawdata` with model parameters
`modelinfo`, algorithm parameters `algparams`, modeling backend `space`, and
a MPI communicator `comm`.

"""
function ProxALEvaluator(
    rawdata::RawData,
    modelinfo::ModelInfo,
    algparams::AlgParams,
    backend::AbstractBackend=JuMPBackend(),
    comm::Union{MPI.Comm,Nothing} = MPI.COMM_WORLD;
    output::Bool=false
)
    opfdata = opf_loaddata(
        rawdata;
        time_horizon_start = modelinfo.time_horizon_start,
        time_horizon_end = modelinfo.time_horizon_start + modelinfo.num_time_periods - 1,
        load_scale = modelinfo.load_scale,
        ramp_scale = modelinfo.ramp_scale,
        corr_scale = modelinfo.corr_scale
    )
    if modelinfo.time_link_constr_type != :penalty
        @warn("ProxAL is guaranteed to converge only when time_link_constr_type = :penalty\n"*
              "         Forcing time_link_constr_type = :penalty\n")
        modelinfo.time_link_constr_type = :penalty
    end
    if modelinfo.num_ctgs > 1 && !algparams.decompCtgs
        if isa(backend, ExaAdmmBackend)
            @warn("ProxAL with multiple contingencies and $backend "*
                  "is guaranteed to converge only when AlgParams.decompCtgs = true\n"*
                  "     Forcing AlgParams.decompCtgs = true\n")
            algparams.decompCtgs = true
        else
            error("Multiple contingencies not supported by backend: $backend")
        end
    end
    if modelinfo.num_ctgs > 1 && algparams.decompCtgs
        if isa(backend, JuMPBackend)
            allowed = [:frequency_penalty, :preventive_penalty, :corrective_penalty]
        elseif isa(backend, AdmmBackend)
            allowed = [:preventive_penalty, :corrective_penalty]
        else
            error("Multiple contingencies not supported by backend: $backend")
        end
        if modelinfo.ctgs_link_constr_type ∉ allowed
            if modelinfo.ctgs_link_constr_type == :frequency_equality
                modelinfo.ctgs_link_constr_type = :frequency_penalty
            elseif modelinfo.ctgs_link_constr_type ∈ [:corrective_equality, :corrective_inequality]
                modelinfo.ctgs_link_constr_type = :corrective_penalty
            else
                modelinfo.ctgs_link_constr_type = :preventive_penalty
            end
            @warn("ProxAL with $backend is guaranteed to converge only when "*
                  "ctgs_link_constr_type in $allowed\n"*
                  "     Forcing ctgs_link_constr_type = $(modelinfo.ctgs_link_constr_type)\n")
        end
    end

    problem = ProxALProblem(opfdata, rawdata, modelinfo, algparams, backend, comm; output=output)
    return ProxALEvaluator(problem, modelinfo, algparams, opfdata, rawdata, backend, comm)
end

"""
    optimize!(nlp::ProxALEvaluator)

Solve problem using the `nlp` evaluator
of the decomposition algorithm.

"""
function optimize!(
    nlp::ProxALEvaluator;
    print_timings::Bool=false,
    θ_t_initial::Union{Real,Nothing}=nothing,
    θ_c_initial::Union{Real,Nothing}=nothing,
    ρ_t_initial::Union{Real,Nothing}=nothing,
    ρ_c_initial::Union{Real,Nothing}=nothing,
    τ_initial::Union{Real,Nothing}=nothing,
    τ_factor::Float64 = 2.0
)
    algparams = nlp.algparams
    modelinfo = nlp.modelinfo
    runinfo   = nlp.problem
    opfdata   = nlp.opfdata
    comm      = nlp.comm

    has_ctgs(modelinfo, algparams) = (algparams.decompCtgs && modelinfo.num_ctgs > 0)
    maxρ(modelinfo, algparams) = has_ctgs(modelinfo, algparams) ? max(algparams.ρ_t, algparams.ρ_c) : algparams.ρ_t
    τ_default(modelinfo, algparams) = τ_factor*maxρ(modelinfo, algparams)

    algparams.θ_t = algparams.θ_c = (1/algparams.tol^2)
    algparams.ρ_t = algparams.ρ_c = modelinfo.obj_scale
    algparams.τ = τ_default(modelinfo, algparams)
    !isnothing(θ_t_initial) && (algparams.θ_t = θ_t_initial)
    !isnothing(θ_c_initial) && (algparams.θ_c = θ_c_initial)
    !isnothing(ρ_t_initial) && (algparams.ρ_t = ρ_t_initial)
    !isnothing(ρ_c_initial) && (algparams.ρ_c = ρ_c_initial)
    !isnothing(τ_initial) && (algparams.τ = τ_initial)
    if (!isnothing(ρ_t_initial) || !isnothing(ρ_c_initial)) && isnothing(τ_initial)
        algparams.τ = τ_default(modelinfo, algparams)
    end

    opfBlockData = runinfo.opfBlockData
    nlp_soltime = runinfo.nlp_soltime
    ngen = length(opfdata.generators)
    nbus = length(opfdata.buses)
    T = modelinfo.num_time_periods
    K = modelinfo.num_ctgs + 1 # base case counted separately
    x = runinfo.x
    λ = runinfo.λ
    # number of contingency per blocks
    k_per_block = (algparams.decompCtgs) ? 1 : K

    function transfer!(blk, x, solution)
        block = opfBlockData.blkIndex[blk]
        k = block[1]
        t = block[2]
        # Pg
        k = algparams.decompCtgs ? k : 1:K
        x.Pg[:,k,t] = solution.pg[:]
        # Qg
        x.Qg[:,k,t] = solution.qg[:]
        # vm
        x.Vm[:,k,t] = solution.vm[:]
        # va
        x.Va[:,k,t] = solution.va[:]
        # wt
        x.ωt[k,t] = algparams.decompCtgs ? solution.ωt[1] : solution.ωt
        # St
        if !algparams.decompCtgs || k == 1
            x.St[:,t] = solution.st[:]
        end
        x.Sk[:,k,t] = solution.sk[:]
    end
    #------------------------------------------------------------------------------------
    function blocknlp_optimize(blk, x_ref, λ_ref, alg_ref)
        model = opfBlockData.blkModel[blk]
        # Update objective
        set_objective!(model, alg_ref, x_ref, λ_ref)
        block = opfBlockData.blkIndex[blk]
        x0 = get_block_view(x_ref, block, modelinfo, algparams)
        solution = optimize!(model, x0, alg_ref)
        transfer!(blk, x, solution)
    end
    #------------------------------------------------------------------------------------
    function blocknlp_init_and_optimize(blk, x_ref, λ_ref, alg_ref)
        model = opfBlockData.blkModel[blk]
        init!(model, alg_ref)
        # Update objective
        set_objective!(model, alg_ref, x_ref, λ_ref)
        block = opfBlockData.blkIndex[blk]
        x0 = get_block_view(x_ref, block, modelinfo, algparams)
        solution = optimize!(model, x0, alg_ref)
        transfer!(blk, x, solution)
    end
    #------------------------------------------------------------------------------------
    function primal_update()
        runinfo.wall_time_elapsed_actual += @elapsed begin
            # Primal update except penalty vars
            nlp_soltime .= 0.0
            nlp_soltime_local = 0.0
            _x = deepcopy(x)
            for blk in runinfo.blkLocalIndices
                nlp_soltime[blk] = @elapsed blocknlp_optimize(blk, _x, λ, algparams)
                nlp_soltime_local += nlp_soltime[blk]
            end
            if comm_rank(comm) == 0
              print_timings && println("Solve subproblems(): $nlp_soltime_local")
            end

            print_timings && comm_barrier(comm)
            elapsed_t = @elapsed begin
                requests = Vector{MPI.Request}()
                requests = vcat(requests, comm_neighbors!(x.Pg, opfBlockData, runinfo, CommPatternTK(), comm))
                requests = vcat(requests, comm_neighbors!(x.Qg, opfBlockData, runinfo, CommPatternTK(), comm))
                requests = vcat(requests, comm_neighbors!(x.Vm, opfBlockData, runinfo, CommPatternTK(), comm))
                requests = vcat(requests, comm_neighbors!(x.Va, opfBlockData, runinfo, CommPatternTK(), comm))
                requests = vcat(requests, comm_neighbors!(x.ωt, opfBlockData, runinfo, CommPatternTK(), comm))
                requests = vcat(requests, comm_neighbors!(x.St, opfBlockData, runinfo, CommPatternT(), comm))
                requests = vcat(requests, comm_neighbors!(x.Sk, opfBlockData, runinfo, CommPatternK(), comm))
                # Every worker sends his contribution
                # comm_sum!(nlp_soltime, comm)
                comm_wait!(requests)
                print_timings && comm_barrier(comm)
            end
            if comm_rank(comm) == 0
                print_timings && println("Comm primals: $elapsed_t")
            end

            elapsed_t = @elapsed begin
                for blk in runinfo.blkLocalIndices
                    update_primal_penalty(x, opfdata, opfBlockData, blk, x, λ, modelinfo, algparams)
                end
                print_timings && comm_barrier(comm)
            end
            if comm_rank(comm) == 0
              print_timings && println("update_primal_penalty(): $elapsed_t")
            end
        end

        print_timings && comm_barrier(comm)
        elapsed_t = @elapsed begin
            requests_zt = comm_neighbors!(x.Zt, opfBlockData, runinfo, CommPatternT(), comm)
            if algparams.decompCtgs
                requests_zk = comm_neighbors!(x.Zk, opfBlockData, runinfo, CommPatternK(), comm)
                comm_wait!(requests_zk)
            end
            comm_wait!(requests_zt)
            print_timings && comm_barrier(comm)
        end
        if comm_rank(comm) == 0
            print_timings && println("Comm penalty: $elapsed_t")
        end
        runinfo.wall_time_elapsed_ideal += isempty(runinfo.blkLinIndices) ? 0.0 : maximum([nlp_soltime[blk] for blk in runinfo.blkLinIndices])
        runinfo.wall_time_elapsed_ideal += elapsed_t
    end
    #------------------------------------------------------------------------------------
    function dual_update()
        elapsed_t = @elapsed begin
            maxviol_t = 0.0; maxviol_c = 0.0
            for blk in runinfo.blkLocalIndices
                lmaxviol_t, lmaxviol_c = update_dual_vars(λ, opfdata, opfBlockData, blk, x, modelinfo, algparams)
                maxviol_t = max(maxviol_t, lmaxviol_t)
                maxviol_c = max(maxviol_c, lmaxviol_c)
            end
            print_timings && comm_barrier(comm)
        end
        if comm_rank(comm) == 0
          print_timings && println("update_dual_vars(): $elapsed_t")
        end
        elapsed_t = @elapsed begin
            requests_ramp = comm_neighbors!(λ.ramping, opfBlockData, runinfo, CommPatternT(), comm)
            if algparams.decompCtgs
                requests_ctgs = comm_neighbors!(λ.ctgs, opfBlockData, runinfo, CommPatternTK(), comm)
                comm_wait!(requests_ctgs)
            end
            comm_wait!(requests_ramp)
            maxviol_t = comm_max(maxviol_t, comm)
            maxviol_c = comm_max(maxviol_c, comm)
            push!(runinfo.maxviol_t, maxviol_t)
            push!(runinfo.maxviol_c, maxviol_c)
            print_timings && comm_barrier(comm)
        end
        if comm_rank(comm) == 0
            print_timings && println("Comm duals: $elapsed_t")
        end
        runinfo.wall_time_elapsed_actual += elapsed_t
        runinfo.wall_time_elapsed_ideal += elapsed_t
    end
    #------------------------------------------------------------------------------------
    function proximal_parameter_update()
        elapsed_t = @elapsed begin
            if algparams.updateτ && runinfo.iter > 1
                delta = (runinfo.lyapunov[end-1] - runinfo.lyapunov[end])/abs(runinfo.lyapunov[end])
                if delta < -1e-4 && algparams.τ < ((2*k_per_block*T) - 1)*maxρ(modelinfo, algparams)
                    algparams.τ *= 2.0
                end
            end
        end
        runinfo.wall_time_elapsed_actual += elapsed_t
        runinfo.wall_time_elapsed_ideal += elapsed_t
    end
    #------------------------------------------------------------------------------------
    function penalty_parameter_update()
        elapsed_t = @elapsed begin
            if max(runinfo.maxviol_t[end], runinfo.maxviol_d[end]) <= algparams.tol
                if runinfo.maxviol_t_actual[end] > algparams.tol && algparams.θ_t < 1e+12
                    algparams.θ_t *= 10.0
                    if algparams.verbose > 0 && comm_rank(comm) == 0
                        (algparams.θ_t >= 1e+12) &&
                            (@warn "penalty parameter too large. problem will likely not converge" maxlog=1)
                    end
                end
            end
            if (algparams.decompCtgs && modelinfo.num_ctgs > 0 &&
                modelinfo.ctgs_link_constr_type ∈ [:preventive_penalty, :corrective_penalty] &&
                max(runinfo.maxviol_c[end], runinfo.maxviol_d[end]) <= algparams.tol
            )
                if runinfo.maxviol_c_actual[end] > algparams.tol && algparams.θ_c < 1e+12
                    algparams.θ_c *= 10.0
                    if algparams.verbose > 0 && comm_rank(comm) == 0
                        (algparams.θ_c >= 1e+12) &&
                            (@warn "penalty parameter too large. problem will likely not converge" maxlog=1)
                    end
                end
            end
        end
        runinfo.wall_time_elapsed_actual += elapsed_t
        runinfo.wall_time_elapsed_ideal += elapsed_t
    end
    #------------------------------------------------------------------------------------
    function auglag_parameter_update()
        elapsed_t = @elapsed begin
            # ρ_t update
            if algparams.updateρ_t
                if runinfo.maxviol_t[end] > 10.0*runinfo.maxviol_d[end] && algparams.ρ_t < 32.0*algparams.θ_t
                    algparams.ρ_t = min(2.0*algparams.ρ_t, 32.0*algparams.θ_t)
                    algparams.τ = τ_default(modelinfo, algparams)
                elseif runinfo.maxviol_d[end] > 10.0*runinfo.maxviol_t[end]
                    algparams.ρ_t = max(0.5*algparams.ρ_t, 1e-4)
                    algparams.τ = τ_default(modelinfo, algparams)
                end
            end

            # ρ_c update
            if algparams.updateρ_c && algparams.decompCtgs && modelinfo.num_ctgs > 0
                @info "tau-update not finalized when decompCtgs = true" maxlog=1
                if runinfo.maxviol_c[end] > 10.0*runinfo.maxviol_d[end] && algparams.ρ_c < 32.0*algparams.θ_c
                    algparams.ρ_c = min(2.0*algparams.ρ_c, 32.0*algparams.θ_c)
                    algparams.τ = τ_default(modelinfo, algparams)
                elseif runinfo.maxviol_d[end] > 10*runinfo.maxviol_c[end]
                    algparams.ρ_c = max(0.5*algparams.ρ_c, 1e-4)
                    algparams.τ = τ_default(modelinfo, algparams)
                end
            end
        end
        runinfo.wall_time_elapsed_actual += elapsed_t
        runinfo.wall_time_elapsed_ideal += elapsed_t
    end

    # Initialization of Pg, Qg, Vm, Va via a OPF solve
    algparams.init_opf && opf_initialization!(nlp)

    function iteration()

        # use this to compute the KKT error at the end of the loop
        runinfo.xprev = deepcopy(x)
        runinfo.λprev = deepcopy(λ)

        # Primal update
        for _ in 1:algparams.num_sweeps
            primal_update()
        end

        # Dual update
        dual_update()

        # Update counters and write output
        runinfo_update(runinfo, opfdata, opfBlockData, modelinfo, algparams, comm)

        # Prox update
        proximal_parameter_update()

        # Penalty update
        penalty_parameter_update()

        # Auglag parameter update
        auglag_parameter_update()
    end


    for runinfo.iter=1:algparams.iterlim
        iteration()

        # Check convergence
        iteration_limit = 0
        for blk in opfBlockData.blkModel
            solution = get_solution(blk)
            if solution.status ∉ MOI_OPTIMAL_STATUSES
                iteration_limit = 1
            end
        end
        giteration_limit = comm_sum(iteration_limit, comm)

        minviol = max(
            runinfo.maxviol_t[end],
            runinfo.maxviol_c[end],
            runinfo.maxviol_t_actual[end],
            runinfo.maxviol_c_actual[end],
            runinfo.maxviol_d[end]
        )

        if minviol < runinfo.minviol
            runinfo.minviol = minviol
            # algparams.tron_outer_eps = minviol
            if runinfo.output
                ProxAL.write(runinfo, nlp, "$(modelinfo.case_name)_$(comm_ranks(comm)).h5")
            end
        end

        if (max(minviol) <= algparams.tol) && (giteration_limit == 0)
        # if minviol <= algparams.tol
            break
        end
    end
    return runinfo
end

function opf_initialization!(nlp::ProxALEvaluator)
    runinfo   = nlp.problem
    modelinfo = nlp.modelinfo
    opfdata   = nlp.opfdata
    rawdata   = nlp.rawdata

    modelinfo_single = deepcopy(modelinfo)
    modelinfo_single.num_time_periods = 1
    primal = ProxAL.OPFPrimalSolution(opfdata, modelinfo_single)
    dual = ProxAL.OPFDualSolution(opfdata, modelinfo_single)
    algparams = AlgParams()
    algparams.mode = :coldstart
    algparams.optimizer = optimizer_with_attributes(Ipopt.Optimizer,
            "print_level" => Int64(algparams.verbose > 0)*5)
    blockmodel = ProxAL.JuMPBlockBackend(1, opfdata, rawdata, algparams, modelinfo_single, 1, 1, 0)
    ProxAL.init!(blockmodel, algparams)
    ProxAL.set_objective!(blockmodel, algparams, primal, dual)
    n = JuMP.num_variables(blockmodel.model)
    x0 = zeros(n)
    solution = ProxAL.optimize!(blockmodel, x0, algparams)
    nbus = length(opfdata.buses)
    ngen = length(opfdata.generators)
    K = modelinfo.num_ctgs
    T = modelinfo.num_time_periods
    for i in 1:T*(K+1)
        for i=1:ngen
            runinfo.x.Pg[i,:,:] .= solution.pg[i]
            runinfo.x.Qg[i,:,:] .= solution.qg[i]
        end
        for i=1:nbus
            runinfo.x.Vm[i,:,:] .= solution.vm[i]
            runinfo.x.Va[i,:,:] .= solution.va[i]
        end
    end
end
