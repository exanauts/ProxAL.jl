using Plots, Measures, DelimitedFiles
using LaTeXStrings

function optionsPlot(plt)
    plot!(plt,
            linewidth = 2.0,
            yscale = :log10,
            framestyle = :box,
            ylim = [1e-6, 1e+2],
            xtickfontsize = 18,
            ytickfontsize = 18,
            guidefontsize = 18,
            titlefontsize = 18,
            legendfontsize = 18,
            size = (800, 800))
end

function initializePlot_iterative()
    gr()
    plt = plot([Inf,Inf], Any[[1,1],[1,1],[1,1],[1,1],[1,1]],
                lab=reshape(
                    [L"||x-x^*||/||x^*||"
                     L"||\displaystyle\Sigma_t A_t x_t - b||"
                     L"||\nabla_x \mathrm{Lagrangian}||"
                     L"|f(x)-f(x^*)|/f(x^*)"
                     L"L-L^*"], 1, 5),
            )
    optionsPlot(plt)
    return plt
end

function updatePlot_iterative(plt, iter, distance, primviol, dualviol, optimgap, delta_lyapunov, savefile = "")
    push!(plt, 1, iter, max(distance, 1e-12))
    push!(plt, 2, iter, max(primviol, 1e-12))
    push!(plt, 3, iter, max(dualviol, 1e-12))
    push!(plt, 4, iter, max(optimgap, 1e-12))
    push!(plt, 5, iter, (delta_lyapunov < 0) ? Inf : max(delta_lyapunov, 1e-12))
    if isempty(savefile)
        savefig("__dummy.png")
    else
        savefig(savefile * ".png")
    end
    #gui()
end

function getDataFilename(prefix::String, case::String, algo::String,
                            num_partitions::Int, sc_constr::Bool, jacobi::Bool, ramp_scale = nothing)
    return prefix * 
            basename(case) * 
            "_" * algo * 
            "_n" * string(num_partitions) * 
            (ramp_scale == nothing ? "" : ("_r" * string(100ramp_scale))) *
            "_f" * string(Int(sc_constr)) * 
            "_j" * string(Int(jacobi)) * 
            ".txt"
end

function getSaveFilename(prefix::String, case::String, algo::String, jacobi::Bool)
    return prefix * 
            basename(case) * 
            "_" * algo * 
            "_j" * string(Int(jacobi)) * 
            ".png"
end

function plot_fixedCaseAlgo(fileprefix::String, savefileprefix::String, case::String, algo::String;
                partitions, other, array2, array2_type::Symbol)
    pyplot()
    p = []
    for (i, ti) in enumerate(partitions)
        for (j, tj) in enumerate(array2)
            if array2_type == :Perturb
                datafile = getDataFilename(fileprefix, case, algo, ti, tj, true, other)
            elseif array2_type == :Ramping
                datafile = getDataFilename(fileprefix, case, algo, ti, other, true, tj)
            else @assert(false)
            end
            savedata = isfile(datafile) ? readdlm(datafile) : 1e-12ones(2000, 4)
            push!(p,
                    plot(savedata[:,1:3], 
                        lab=["Distance to optimal"  "Primal infeasibility"  "Dual infeasibility"])
            )
            optionsPlot(p[end])
            (j == 1) && plot!(p[end], ylabel="Partitions = "*string(ti))
            (i == 1) && plot!(p[end], title=string(array2_type)*" = "*string(100tj)*"%")
            (i == length(partitions)) && plot!(p[end], xlabel="Iterations")
        end
    end
    if !isempty(p)
        plot(p..., layout = (length(partitions), length(array2)), size = (1000, 1000))
        savefig(getSaveFilename(savefileprefix, case, algo, true))
    end
end

function plot_fixedPartitions(fileprefix::String, savefileprefix::String, num_partitions;
                case, algo, other1, other2, other2_type::Symbol)
    pyplot()
    p = []
    for (i, ti) in enumerate(case)
        for (j, tj) in enumerate(algo)
            if other2_type == :Perturb
                datafile = getDataFilename(fileprefix, ti, tj, num_partitions, other2, true, other1)
            elseif other2_type == :Ramping
                datafile = getDataFilename(fileprefix, ti, tj, num_partitions, other1, true, other2)
            else @assert(false)
            end
            savedata = isfile(datafile) ? readdlm(datafile) : 1e-12ones(2000, 4)
            push!(p,
                    plot(savedata[:,1:4],
                        lab=["Distance to optimal"  "Primal infeasibility"  "Dual infeasibility" "Optimality gap"])
            )
            optionsPlot(p[end])
            (j == 1) && plot!(p[end], ylabel=string(ti))
            (i == 1) && plot!(p[end], title=string(tj))
        end
    end
    if !isempty(p)
        plot(p..., layout = (length(case), length(algo)), size = (1000, 1000))
        savefig(getSaveFilename(savefileprefix, "T" * string(num_partitions), string(other2_type) * string(other2), true))
    end
end

