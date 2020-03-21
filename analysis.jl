using Plots, Measures, DelimitedFiles

function optionsPlot(plt)
    plot!(plt,
            linewidth = 2.0,
            linealpha = 0.7,
            yscale = :log10,
            framestyle = :box,
            ylim = [1e-5, 1e1],
            xtickfontsize = 12,
            ytickfontsize = 12,
            guidefontsize = 14,
            titlefontsize = 14,
            legendfontsize = 10,
            size = (1000, 1000))
end

function initializePlot_iterative()
    gr()
    plt = plot([Inf,Inf], Any[[1,1],[1,1],[1,1],[1,1]],
                lab=["Distance to optimal"  "Primal infeasibility"  "Dual infeasibility" "Optimality gap"])
    optionsPlot(plt)
    return plt
end

function updatePlot_iterative(plt, iter, distance, primviol, dualviol, optimgap)#, xavg_primviol=nothing, xavg_dualviol=nothing)
    push!(plt, 1, iter, distance)
    push!(plt, 2, iter, primviol)
    push!(plt, 3, iter, dualviol)
    push!(plt, 4, iter, optimgap)
    #push!(plt, 4, iter, xavg_primviol)
    #push!(plt, 5, iter, xavg_dualviol)
    gui()
end

function getDataFilename(prefix::String, case::String, algo::String,
                            num_partitions::Int, perturbation::Number, jacobi::Bool, ramp_scale = nothing)
    return prefix * 
            basename(case) * 
            "_" * algo * 
            "_n" * string(num_partitions) * 
            (ramp_scale == nothing ? "" : ("_r" * string(100ramp_scale))) *
            "_f" * string(perturbation) * 
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
