using Plots

function convergence_plot(iter, maxviol_t, maxviol_d, dist_x, optimgap, lyapunov_gap, detailed_plot = false)
    ENV["GKSwstype"]="nul"
    gr()
    if detailed_plot
        plt1 = plot(1:iter,
                    [dist_x maxviol_t maxviol_d optimgap lyapunov_gap],
                    lab = ["|x - x_optimal|" "Ramping error" "KKT error" "Objective function error" "Lyapunov function value"])
    else
        plt1 = plot(1:iter,
                    [maxviol_t maxviol_d],
                    lab = ["Ramping error" "KKT error"])
    end
    plot!(plt1, lw = 2.0,
                xlabel = "Iteration",
                ylabel = "Error metric",
                yscale = :log10,
                framestyle = :box,
                ylim = [1e-4, 1e+1],
                xlim = [0, 100],
                xtickfontsize = 20,
                ytickfontsize = 20,
                guidefontsize = 20,
                titlefontsize = 20,
                legendfontsize = 20,
                size = (800, 800))
    return plt1
end
