struct ProxALEvaluator <: AbstractNLPEvaluator
    alminfo::ProxALMData
    modelinfo::ModelParams
    algparams::AlgParams
    opfdata::OPFData
    rawdata::RawData
    space::AbstractSpace
    comm::MPI.Comm
end

"""
    ProxALEvaluator(case_file::String, load_file::String,
                modelinfo::ModelParams,
                algparams::AlgParams,
                space::AbstractSpace=JuMPBackend(),
                comm::MPI.Comm = MPI.COMM_WORLD)

Instantiate multi-period ACOPF
specified in `case_file` with loads in `load_file` with model parameters
`modelinfo` and algorithm parameters `algparams`, and
a MPI communicator `comm`.
"""
function ProxALEvaluator(
    case_file::String, load_file::String,
    modelinfo::ModelParams,
    algparams::AlgParams,
    space::AbstractSpace=JuMPBackend(),
    opt_sol::Dict = Dict(),
    lyapunov_sol::Dict = Dict(),
    comm::MPI.Comm = MPI.COMM_WORLD
)
    rawdata = RawData(case_file, load_file)
    opfdata = opf_loaddata(
        rawdata;
        time_horizon_start = 1,
        time_horizon_end = modelinfo.num_time_periods,
        load_scale = modelinfo.load_scale,
        ramp_scale = modelinfo.ramp_scale
    )
    set_penalty!(
        algparams,
        length(opfdata.generators),
        modelinfo.maxρ_t,
        modelinfo.maxρ_c,
        modelinfo
    )

    # ctgs_arr = deepcopy(rawdata.ctgs_arr)
    alminfo = ProxALMData(opfdata, rawdata, modelinfo, algparams, space, opt_sol, lyapunov_sol)
    return ProxALEvaluator(alminfo, modelinfo, algparams, opfdata, rawdata, space, comm)
end

function optimize!(nlp::ProxALEvaluator)
    algparams = nlp.algparams
    modelinfo = nlp.modelinfo
    runinfo   = nlp.alminfo
    opfdata   = nlp.opfdata
    comm      = nlp.comm

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
        opt_sol[fr:to, blk] .= solution.pg[:]
        # Qg
        fr = to + 1 ; to += ngen * k_per_block
        opt_sol[fr:to, blk] .= solution.qg[:]
        # vm
        fr = to +1 ; to = fr + nbus * k_per_block - 1
        opt_sol[fr:to, blk] .= solution.vm[:]
        # va
        fr = to + 1 ; to = fr + nbus * k_per_block - 1
        opt_sol[fr:to, blk] .= solution.va[:]
        # wt
        fr = to +1  ; to = fr + k_per_block -1
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

    # Initialization of Pg, Qg, Vm, Va via a OPF solve
    modelinfo.init_opf && opf_initialization!(nlp)


    for runinfo.iter=1:algparams.iterlim

        if runinfo.initial_solve
            if runinfo.iter == 1
                set_penalty!(algparams,
                         length(opfdata.generators),
                         0.0,
                         0.0,
                         modelinfo)
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

function opf_initialization!(nlp::ProxALEvaluator)
    runinfo   = nlp.alminfo
    modelinfo = nlp.modelinfo
    opfdata   = nlp.opfdata
    rawdata   = nlp.rawdata

    modelinfo_single = deepcopy(modelinfo)
    modelinfo_single.num_time_periods = 1
    primal = ProxAL.PrimalSolution(opfdata, modelinfo_single)
    dual = ProxAL.DualSolution(opfdata, modelinfo_single)
    algparams = AlgParams()
    algparams.parallel = false #algparams.parallel = (nprocs() > 1)
    algparams.mode = :coldstart
    algparams.optimizer = optimizer_with_attributes(Ipopt.Optimizer,
            "print_level" => Int64(algparams.verbose > 0)*5)
    blockmodel = ProxAL.JuMPBlockModel(1, opfdata, rawdata, algparams, modelinfo_single, 1, 1, 0)
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
