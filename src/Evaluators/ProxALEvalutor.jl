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
        space::AbstractBackend=JuMPBackend(),
        comm::MPI.Comm = MPI.COMM_WORLD
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
    opt_sol::Dict = Dict(),
    lyapunov_sol::Dict = Dict(),
    comm::Union{MPI.Comm,Nothing} = MPI.COMM_WORLD
)
    rawdata = RawData(case_file, load_file)
    opfdata = opf_loaddata(
        rawdata;
        time_horizon_start = 1,
        time_horizon_end = modelinfo.num_time_periods,
        load_scale = modelinfo.load_scale,
        ramp_scale = modelinfo.ramp_scale
    )
    if modelinfo.time_link_constr_type != :penalty
        @warn("ProxAL is guaranteed to converge only when time_link_constr_type = :penalty\n"*
              "         Forcing time_link_constr_type = :penalty\n")
        modelinfo.time_link_constr_type = :penalty
    end
    if modelinfo.num_ctgs > 1 && algparams.decompCtgs
        if modelinfo.ctgs_link_constr_type ∉ [:frequency_penalty, :preventive_penalty, :corrective_penalty]
            str = "ProxAL is guaranteed to converge only when "*
                  "ctgs_link_constr_type ∈ [:frequency_penalty, :preventive_penalty, :corrective_penalty]\n"
            if modelinfo.ctgs_link_constr_type == :preventive_equality
                @warn(str * "         Forcing ctgs_link_constr_type = :preventive_penalty\n")
                modelinfo.ctgs_link_constr_type = :preventive_penalty
            elseif modelinfo.ctgs_link_constr_type ∈ [:corrective_equality, :corrective_inequality]
                @warn(str * "         Forcing ctgs_link_constr_type = :corrective_penalty\n")
                modelinfo.ctgs_link_constr_type = :corrective_penalty
            else
                @warn(str * "         Forcing ctgs_link_constr_type = :frequency_penalty\n")
                modelinfo.ctgs_link_constr_type = :frequency_penalty
            end
        end
    end

    # ctgs_arr = deepcopy(rawdata.ctgs_arr)
    problem = ProxALProblem(opfdata, rawdata, modelinfo, algparams, space, comm, opt_sol, lyapunov_sol)
    return ProxALEvaluator(problem, modelinfo, algparams, opfdata, rawdata, space, comm)
end

"""
    optimize!(nlp::ProxALEvaluator)

Solve problem using the `nlp` evaluator
of the decomposition algorithm.

"""
function optimize!(nlp::ProxALEvaluator; print_timings=false)
    algparams = nlp.algparams
    modelinfo = nlp.modelinfo
    runinfo   = nlp.problem
    opfdata   = nlp.opfdata
    comm      = nlp.comm

    algparams.θ_t = algparams.θ_c = (1/algparams.tol^2)
    algparams.ρ_t = algparams.ρ_c = modelinfo.obj_scale
    algparams.τ = 2.0*max(algparams.ρ_t, algparams.ρ_c)
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
    # number of contingency per blocks
    k_per_block = (algparams.decompCtgs) ? 1 : K

    function transfer!(blk, opt_sol, solution)
        # Pg
        fr = 1 ; to = ngen * k_per_block
        @views opt_sol[fr:to, blk] .= solution.pg[:]
        # Qg
        fr = to + 1 ; to += ngen * k_per_block
        @views opt_sol[fr:to, blk] .= solution.qg[:]
        # vm
        fr = to +1 ; to = fr + nbus * k_per_block - 1
        @views opt_sol[fr:to, blk] .= solution.vm[:]
        # va
        fr = to + 1 ; to = fr + nbus * k_per_block - 1
        @views opt_sol[fr:to, blk] .= solution.va[:]
        # wt
        fr = to +1  ; to = fr + k_per_block -1
        @views opt_sol[fr:to, blk] .= solution.ωt[:]
        # St
        fr = to +1  ; to = fr + ngen - 1
        @views opt_sol[fr:to, blk] .= solution.st[:]
        # zt
        fr = to +1  ; to = fr + ngen -1
        if haskey(solution, :zt)
            @views opt_sol[fr:to, blk] .= solution.zt[:]
        end
        # sk
        fr = to +1  ; to = fr + ngen * k_per_block - 1
        if haskey(solution, :sk)
            @views opt_sol[fr:to, blk] .= solution.sk[:]
        end
        # zk
        fr = to +1  ; to = fr + ngen * k_per_block - 1
        if haskey(solution, :zk)
            @views opt_sol[fr:to, blk] .= solution.zk[:]
        end
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
            nlp_soltime .= 0.0
            nlp_soltime_local = 0.0
            for blk in runinfo.par_order
                if ismywork(blk, comm)
                    nlp_opt_sol[:,blk] .= 0.0
                    # nlp_soltime[blk] = @elapsed blocknlp_copy(blk; x_ref = x, λ_ref = λ, alg_ref = algparams)
                    nlp_soltime[blk] = @elapsed blocknlp_recreate(blk, x, λ, algparams)
                    nlp_soltime_local += nlp_soltime[blk]
                end
            end
            if comm_rank(comm) == 0
              print_timings && println("Solve subproblems(): $nlp_soltime_local")
            end

            print_timings && comm_barrier(comm)
            elapsed_t = @elapsed begin
                requests = comm_neighbors!(nlp_opt_sol, opfBlockData, runinfo, CommPatternTK(), comm)
                # Every worker sends his contribution
                comm_sum!(nlp_soltime, comm)
                comm_wait!(requests)
                print_timings && comm_barrier(comm)
            end
            if comm_rank(comm) == 0
                print_timings && println("Comm primals: $elapsed_t")
            end

            # Update primal values
            elapsed_t = @elapsed begin
                for blk in runinfo.par_order
                    block = opfBlockData.blkIndex[blk]
                    k = block[1]
                    t = block[2]
                    if ismywork(blk, comm)
                        # Updating my own primal values
                        opfBlockData.colValue[:,blk] .= nlp_opt_sol[:,blk]
                        update_primal_nlpvars(x, opfBlockData, blk, modelinfo, algparams)
                        for blkn in runinfo.par_order
                            blockn = opfBlockData.blkIndex[blkn]
                            kn = blockn[1]
                            tn = blockn[2]
                            # Updating the received neighboring primal values
                            if ((tn == t-1 || tn == t+1) && kn == 1) && !ismywork(blkn, comm)
                                opfBlockData.colValue[:,blkn] .= nlp_opt_sol[:,blkn]
                                update_primal_nlpvars(x, opfBlockData, blkn, modelinfo, algparams)
                            end
                            if (tn == t) && !ismywork(blkn, comm)
                                opfBlockData.colValue[:,blkn] .= nlp_opt_sol[:,blkn]
                                update_primal_nlpvars(x, opfBlockData, blkn, modelinfo, algparams)
                            end
                        end
                    end
                end
                print_timings && comm_barrier(comm)
            end

            if comm_rank(comm) == 0
              print_timings && println("update_primal_nlpvars(): $elapsed_t")
            end


            # Primal update of penalty vars
            elapsed_t = @elapsed begin
                for blk in runinfo.par_order
                    if ismywork(blk, comm)
                        update_primal_penalty(x, opfdata, opfBlockData, blk, x, λ, modelinfo, algparams)
                    end
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
            requests_zk = comm_neighbors!(x.Zk, opfBlockData, runinfo, CommPatternK(), comm)
            comm_wait!(requests_zt)
            comm_wait!(requests_zk)
            print_timings && comm_barrier(comm)
        end
        if comm_rank(comm) == 0
            print_timings && println("Comm penalty: $elapsed_t")
        end
        runinfo.wall_time_elapsed_ideal += isempty(runinfo.par_order) ? 0.0 : maximum([nlp_soltime[blk] for blk in runinfo.par_order])
        runinfo.wall_time_elapsed_ideal += elapsed_t
    end
    #------------------------------------------------------------------------------------
    function dual_update()
        elapsed_t = @elapsed begin
            maxviol_t = 0.0; maxviol_c = 0.0
            for blk in runinfo.par_order
                block = opfBlockData.blkIndex[blk]
                k = block[1]
                t = block[2]
                if ismywork(blk, comm)
                    lmaxviol_t, lmaxviol_c = update_dual_vars(λ, opfdata, opfBlockData, blk, x, modelinfo, algparams)
                    maxviol_t = max(maxviol_t, lmaxviol_t)
                    maxviol_c = max(maxviol_c, lmaxviol_c)
                end
            end
            print_timings && comm_barrier(comm)
        end
        if comm_rank(comm) == 0
          print_timings && println("update_dual_vars(): $elapsed_t")
        end
        elapsed_t = @elapsed begin
            requests_ramp = comm_neighbors!(λ.ramping, opfBlockData, runinfo, CommPatternT(), comm)
            requests_ctgs = comm_neighbors!(λ.ctgs, opfBlockData, runinfo, CommPatternK(), comm)
            comm_wait!(requests_ramp)
            comm_wait!(requests_ctgs)
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
                maxθ = algparams.decompCtgs  ? max(algparams.θ_t, algparams.θ_c) : algparams.θ_t
                delta = (runinfo.lyapunov[end-1] - runinfo.lyapunov[end])/abs(runinfo.lyapunov[end])
                if delta < -1e-4 && algparams.τ < 320.0*maxθ
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
                    algparams.τ = algparams.decompCtgs ? 2.0*max(algparams.ρ_t, algparams.ρ_c) : 2.0*algparams.ρ_t
                elseif runinfo.maxviol_d[end] > 10.0*runinfo.maxviol_t[end]
                    algparams.ρ_t *= 0.5
                    algparams.τ = algparams.decompCtgs ? 2.0*max(algparams.ρ_t, algparams.ρ_c) : 2.0*algparams.ρ_t
                end
            end

            # ρ_c update
            if algparams.updateρ_c && algparams.decompCtgs && modelinfo.num_ctgs > 0
                @info "tau-update not finalized when decompCtgs = true" maxlog=1
                if runinfo.maxviol_c[end] > 10.0*runinfo.maxviol_d[end] && algparams.ρ_c < 32.0*algparams.θ_c
                    algparams.ρ_c = min(2.0*algparams.ρ_c, 32.0*algparams.θ_c)
                    algparams.τ = 2.0*max(algparams.ρ_t, algparams.ρ_c)
                elseif runinfo.maxviol_d[end] > 10*runinfo.maxviol_c[end]
                    algparams.ρ_c *= 0.5
                    algparams.τ = 2.0*max(algparams.ρ_t, algparams.ρ_c)
                end
            end
        end
        runinfo.wall_time_elapsed_actual += elapsed_t
        runinfo.wall_time_elapsed_ideal += elapsed_t
    end

    # Initialization of Pg, Qg, Vm, Va via a OPF solve
    algparams.init_opf && opf_initialization!(nlp)

    function iteration()

        if runinfo.initial_solve
            if runinfo.iter == 1
                algparams.ρ_t = 0.0
                algparams.ρ_c = 0.0
                algparams.τ = 0.0
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
        if max(
            runinfo.maxviol_t[end],
            runinfo.maxviol_c[end],
            runinfo.maxviol_t_actual[end],
            runinfo.maxviol_c_actual[end],
            runinfo.maxviol_d[end]
        ) <= algparams.tol
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
