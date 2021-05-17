mutable struct ProxALMData
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
    ser_order
    par_order
    plt

    function ProxALMData(
        opfdata::OPFData, rawdata::RawData,
        modelinfo::ModelParams,
        algparams::AlgParams,
        space::AbstractSpace,
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
        # NLP blocks
        blocks = OPFBlocks(
            opfdata, rawdata;
            modelinfo=modelinfo, algparams=algparams,
            backend=(space==FullSpace()) ? JuMPBlockModel : ExaBlockModel,
        )

        blkLinIndex = LinearIndices(blocks.blkIndex)
        for blk in blkLinIndex
            model = blocks.blkModel[blk]
            init!(model, algparams)
            blocks.colValue[:,blk] .= get_block_view(x, blocks.blkIndex[blk], modelinfo, algparams)
        end

        ser_order = blkLinIndex
        par_order = []
        if algparams.parallel
            if algparams.jacobi
                ser_order = []
                par_order = blkLinIndex
            elseif algparams.decompCtgs
                ser_order = @view blkLinIndex[1,:]
                par_order = @view blkLinIndex[2:end,:]
            end
        end
        plt = nothing
        iter = 0
        objvalue = []
        lyapunov = []
        maxviol_t = []
        maxviol_c = []
        maxviol_d = []
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
            ser_order,
            par_order,
            plt
        )
    end
end

function update_runinfo(
    runinfo::ProxALMData, opfdata::OPFData,
    modelinfo::ModelParams,
    algparams::AlgParams
)
    iter = runinfo.iter
    push!(
        runinfo.objvalue,
        compute_objective_function(runinfo.x, opfdata, modelinfo)
    )
    push!(
        runinfo.lyapunov,
        compute_lyapunov_function(runinfo.x, runinfo.λ, opfdata, runinfo.xprev, modelinfo, algparams)
    )
    push!(
        runinfo.maxviol_d,
        compute_dual_error(runinfo.x, runinfo.xprev, runinfo.λ, runinfo.λprev, opfdata, modelinfo, algparams)
    )
    push!(runinfo.dist_x, NaN)
    push!(runinfo.dist_λ, NaN)
    optimgap = NaN
    lyapunov_gap = NaN
    if !isempty(runinfo.opt_sol)
        xstar = runinfo.opt_sol["primal"]
        λstar = runinfo.opt_sol["dual"]
        zstar = runinfo.opt_sol["objective_value_nondecomposed"]
        runinfo.dist_x[iter] = dist(runinfo.x, xstar, modelinfo, algparams)
        runinfo.dist_λ[iter] = dist(runinfo.λ, λstar, modelinfo, algparams)
        optimgap = 100.0abs(runinfo.objvalue[iter] - zstar)/abs(zstar)
    end
    if !isempty(runinfo.lyapunov_sol)
        lyapunov_star = runinfo.lyapunov_sol["objective_value_lyapunov_bound"]
        lyapunov_gap = 100.0(runinfo.lyapunov[end] - lyapunov_star)/abs(lyapunov_star)
    end

    if algparams.verbose > 0
        @printf("iter %3d: ramp_err = %.3f, ctgs_err = %.3f, dual_err = %.3f, |x-x*| = %.3f, |λ-λ*| = %.3f, gap = %.2f%%, lyapgap = %.2f%%\n",
                    iter,
                    runinfo.maxviol_t[iter],
                    runinfo.maxviol_c[iter],
                    runinfo.maxviol_d[iter],
                    runinfo.dist_x[iter],
                    runinfo.dist_λ[iter],
                    optimgap,
                    lyapunov_gap)
    end
end
