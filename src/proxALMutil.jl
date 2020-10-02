using Plots, LaTeXStrings

mutable struct ProxALMData
    opfBlockData::OPFBlockData
    x::PrimalSolution
    λ::DualSolution
    xprev::PrimalSolution
    λprev::DualSolution
    lyapunovprev::Float64
    maxviol_t::Float64
    maxviol_c::Float64
    maxviol_d::Float64
    opt_sol::Dict
    lyapunov_sol::Dict
    initial_solve::Bool
    nlp_opt_sol::SharedArray{Float64,2}
    nlp_soltime::SharedVector{Float64}
    wall_time_elapsed_actual::Float64
    wall_time_elapsed_ideal::Float64
    ser_order
    par_order
    iter
    plt

    function ProxALMData(opfdata::OPFData, rawdata::RawData, optimizer;     
                         modelinfo::ModelParams,
                         algparams::AlgParams,
                         fullmodel::Bool = false,
                         initial_primal = nothing,
                         initial_dual = nothing)
        optfile_nondecomposed = modelinfo.savefile * ".nondecomposed.jld"
        optfile_lyapunov = modelinfo.savefile * ".lyapunov.jld"
        lyapunov_sol, opt_sol = Dict(), Dict()
        if fullmodel
            mode_oldval = algparams.mode
            algparams.mode = :nondecomposed
            opt_sol = solve_fullmodel(opfdata, rawdata, optimizer; modelinfo = modelinfo, algparams = algparams)
            algparams.mode = :lyapunov_bound
            lyapunov_sol = solve_fullmodel(opfdata, rawdata, optimizer; modelinfo = modelinfo, algparams = algparams)
            algparams.mode = mode_oldval

            if algparams.verbose > 2
                JLD.save(optfile_nondecomposed, opt_sol)
                JLD.save(optfile_lyapunov, lyapunov_sol)
            end
        else
            if isfile(optfile_nondecomposed)
                opt_sol = JLD.load(optfile_nondecomposed)
            end
            if isfile(optfile_lyapunov)
                lyapunov_sol = JLD.load(optfile_lyapunov)
            end
        end
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
                PrimalSolution(opfdata; modelinfo = modelinfo) :
                deepcopy(initial_primal)
        λ = (initial_dual === nothing) ?
                DualSolution(opfdata; modelinfo = modelinfo) :
                deepcopy(initial_dual)
        # NLP blocks
        opfBlockData = OPFBlockData(opfdata, rawdata, optimizer; modelinfo = modelinfo, algparams = algparams)
        blkLinIndex = LinearIndices(opfBlockData.blkIndex)
        for blk in blkLinIndex
            opfBlockData.blkModel[blk] = opf_block_model_initialize(blk, opfBlockData, rawdata; algparams = algparams)
            opfBlockData.colValue[:,blk] .= get_block_view(x, opfBlockData.blkIndex[blk]; modelinfo = modelinfo, algparams = algparams)
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
        plt = (algparams.verbose > 1) ? initialize_plot() : nothing
        iter = 0
        lyapunovprev = Inf
        maxviol_t = Inf
        maxviol_c = Inf
        maxviol_d = Inf
        nlp_opt_sol = SharedArray{Float64, 2}(opfBlockData.colCount, opfBlockData.blkCount)
        nlp_opt_sol .= opfBlockData.colValue
        nlp_soltime = SharedVector{Float64}(opfBlockData.blkCount)
        wall_time_elapsed_actual = 0.0
        wall_time_elapsed_ideal = 0.0
        xprev = deepcopy(x)
        λprev = deepcopy(λ)



        new(opfBlockData,
            x,
            λ,
            xprev,
            λprev,
            lyapunovprev,
            maxviol_t,
            maxviol_c,
            maxviol_d,
            opt_sol,
            lyapunov_sol,
            initial_solve,
            nlp_opt_sol,
            nlp_soltime,
            wall_time_elapsed_actual,
            wall_time_elapsed_ideal,
            ser_order,
            par_order,
            iter,
            plt)
    end
end

function print_runinfo(runinfo::ProxALMData, opfdata::OPFData;
                       modelinfo::ModelParams,
                       algparams::AlgParams)
    objvalue = compute_objective_function(runinfo.x, opfdata; modelinfo = modelinfo)
    lyapunov = compute_lyapunov_function(runinfo.x, runinfo.λ, opfdata; xref = runinfo.xprev, modelinfo = modelinfo, algparams = algparams)
    runinfo.maxviol_d =
        compute_dual_error(runinfo.x, runinfo.xprev, runinfo.λ, runinfo.λprev, opfdata; modelinfo = modelinfo, algparams = algparams)
    dist_x = NaN
    dist_λ = NaN
    optimgap = NaN
    lyapunov_gap = NaN
    if !isempty(runinfo.opt_sol)
        xstar = runinfo.opt_sol["primal"]
        λstar = runinfo.opt_sol["dual"]
        zstar = runinfo.opt_sol["objective_value_nondecomposed"]
        dist_x = dist(runinfo.x, xstar; modelinfo = modelinfo, algparams = algparams)
        dist_λ = dist(runinfo.λ, λstar; modelinfo = modelinfo, algparams = algparams)
        optimgap = 100.0abs(objvalue - zstar)/abs(zstar)
    end
    if !isempty(runinfo.lyapunov_sol)
        lyapunov_star = runinfo.lyapunov_sol["objective_value_lyapunov_bound"]
        lyapunov_gap = 100.0(lyapunov - lyapunov_star)/abs(lyapunov_star)
    end

    if algparams.verbose > 0
        @printf("iter %3d: ramp_err = %.3f, ctgs_err = %.3f, dual_err = %.3f, |x-x*| = %.3f, |λ-λ*| = %.3f, gap = %.2f%%, lyapgap = %.2f%%\n",
                    runinfo.iter,
                    runinfo.maxviol_t,
                    runinfo.maxviol_c,
                    runinfo.maxviol_d,
                    dist_x,
                    dist_λ,
                    optimgap,
                    lyapunov_gap)
    end

    if algparams.verbose > 1
        plotzero = algparams.zero
        iter = runinfo.iter
        push!(runinfo.plt, 1, iter, max(runinfo.maxviol_t, plotzero))
        push!(runinfo.plt, 2, iter, max(runinfo.maxviol_c, plotzero))
        push!(runinfo.plt, 3, iter, max(runinfo.maxviol_d, plotzero))
        push!(runinfo.plt, 4, iter, max(dist_x, plotzero))
        push!(runinfo.plt, 5, iter, max(dist_λ, plotzero))
        push!(runinfo.plt, 6, iter, max(optimgap, plotzero))
        push!(runinfo.plt, 7, iter, max(lyapunov_gap, plotzero))
        savefig(runinfo.plt, modelinfo.savefile * ".plot.png")
    end

    if algparams.verbose > 2
        savefile = modelinfo.savefile * ".iter_" * string(runinfo.iter) * ".jld"
        iter_sol = Dict()
        iter_sol["x"] = x
        iter_sol["λ"] = λ
        JLD.save(savefile, iter_sol)
    end
end

function options_plot(plt)
    fsz = 20
    plot!(plt,
          fontfamily = "Computer-Modern",
          yscale = :log10,
          framestyle = :box,
          ylim = [1e-4, 1e+1],
          xtickfontsize = fsz,
          ytickfontsize = fsz,
          guidefontsize = fsz,
          titlefontsize = fsz,
          legendfontsize = fsz,
          size = (800, 800)
    )
end

function initialize_plot()
    gr()
    label = [L"|p^0_{gt} - p^0_{g,t-1}| - r_g"
             L"|p^k_{gt} - p^0_{gt} - \alpha_g \omega^k_t|"
             L"|\textrm{\sffamily KKT}|"
             L"|x-x^*|"
             L"|\lambda-\lambda^*|"
             L"|c(x)-c(x^*)|/c(x^*)"
             L"|L-L^*|/L^*"]
    any = Array{Any, 1}(undef, length(label))
    any .= Any[[1,1]]
    plt = plot([Inf, Inf], any,
                lab = reshape(label, 1, length(label)),
                lw = 2.5,
                # markersize = 2.5,
                # markershape = :auto,
                xlabel=L"\textrm{\sffamily Iteration}")
    options_plot(plt)
    return plt
end
