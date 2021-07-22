mutable struct ProxALProblem
    opfBlockData::OPFBlocks

    #---- iterate information ----
    x::PrimalSolution
    λ::DualSolution
    xprev::PrimalSolution
    λprev::DualSolution
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
                PrimalSolution(opfdata, modelinfo) :
                deepcopy(initial_primal)
        λ = (initial_dual === nothing) ?
                DualSolution(opfdata, modelinfo) :
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
            if ismywork(blk, comm)
                model = blocks.blkModel[blk]
                init!(model, algparams)
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

        new(
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
end

function runinfo_update(
    runinfo::ProxALProblem, opfdata::OPFData,
    opfBlockData::OPFBlocks,
    modelinfo::ModelInfo,
    algparams::AlgParams,
    comm::Union{MPI.Comm,Nothing}
)
    iter = runinfo.iter
    obj = 0.0
    for blk in runinfo.par_order
        if ismywork(blk, comm)
            obj += compute_objective_function(runinfo.x, opfdata, opfBlockData, blk, modelinfo, algparams)
        end
    end
    obj = comm_sum(obj, comm)
    push!(runinfo.objvalue, obj)

    iter = runinfo.iter
    lyapunov = 0.0
    for blk in runinfo.par_order
        if ismywork(blk, comm)
            lyapunov += compute_lyapunov_function(runinfo.x, runinfo.λ, opfdata, opfBlockData, blk, runinfo.xprev, modelinfo, algparams)
        end
    end
    lyapunov = comm_sum(lyapunov, comm)
    push!(runinfo.lyapunov, lyapunov)

    maxviol_t_actual = 0.0
    for blk in runinfo.par_order
        if ismywork(blk, comm)
            maxviol_t_actual = compute_true_ramp_error(runinfo.x, opfdata, opfBlockData, blk, modelinfo)
        end
    end
    maxviol_t_actual = comm_max(maxviol_t_actual, comm)
    push!(runinfo.maxviol_t_actual, maxviol_t_actual)

    maxviol_c_actual = 0.0
    for blk in runinfo.par_order
        if ismywork(blk, comm)
            maxviol_c_actual = compute_true_ctgs_error(runinfo.x, opfdata, opfBlockData, blk, modelinfo)
        end
    end
    maxviol_c_actual = comm_max(maxviol_c_actual, comm)
    push!(runinfo.maxviol_c_actual, maxviol_c_actual)

    # FIX ME: Frigging bug in the parallel implementation of the dual error
    # FIX ME: Re-implement parallel implementation of the dual error
    # (ngen, K, T) = size(runinfo.x.Pg)
    # smaxviol_d = 3*ngen*K + 2*ngen + K
    # @show smaxviol_d
    # maxviol_d = zeros(Float64, smaxviol_d)
    # for blk in runinfo.par_order
    #     if ismywork(blk, comm)
    #         maxviol_d .+= compute_dual_error(runinfo.x, runinfo.xprev, runinfo.λ, runinfo.λprev, opfdata, opfBlockData, blk, modelinfo, algparams)
    #         # compute_dual_error(runinfo.x, runinfo.xprev, runinfo.λ, runinfo.λprev, opfdata, opfBlockData, blk, modelinfo, algparams)
    #     end
    # end
    # maxviol_d = MPI.Allreduce(maxviol_d, MPI.SUM, comm)
    # maxviol_d = norm(maxviol_d, Inf)
    # push!(runinfo.maxviol_d, maxviol_d)
    maxviol_d = compute_dual_error(runinfo.x, runinfo.xprev, runinfo.λ, runinfo.λprev, opfdata, modelinfo, algparams)
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
            @printf("---------------------------------------------------------------------------------------------------------------\n");
            @printf("iter ramp_err   ramp_err   ctgs_err   ctgs_err   dual_error lyapunov_f   rho_t   rho_c theta_t theta_c     tau \n");
            @printf("     (penalty)  (actual)   (penalty)  (actual)\n");
            @printf("---------------------------------------------------------------------------------------------------------------\n");
        end
        @printf("%4d ", iter-1);
        @printf("%10.4e ", runinfo.maxviol_t[iter])
        @printf("%10.4e ", runinfo.maxviol_t_actual[iter])
        @printf("%10.4e ", runinfo.maxviol_c[iter])
        @printf("%10.4e ", runinfo.maxviol_c_actual[iter])
        @printf("%10.4e ", runinfo.maxviol_d[iter])
        @printf("%10.4e ", runinfo.lyapunov[iter])
        @printf("%7.2f ", algparams.ρ_t)
        @printf("%7.2f ", algparams.ρ_c)
        @printf("%7.2f ", algparams.θ_t)
        @printf("%7.2f ", algparams.θ_c)
        @printf("%7.2f ", algparams.τ)
        @printf("\n")
    end
end
