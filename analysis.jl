using Plots, Measures, DelimitedFiles

function initializePlot_iterative()
    gr()
    return plot([Inf,Inf], Any[[1,1],[1,1],[1,1],[1,1],[1,1]],
                lw = 2,
                linealpha = 0.7,
                xlim = [1, 2000],
                ylim = [1e-5, 1e1],
                framestyle = :box,
                size = (800,800),
                yscale = :log10,
                lab=["distance"  "primviol"  "dualviol" "prinfeas" "KKTerror"])
end

function updatePlot_iterative(plt, iter, dist, primviol, dualviol, primfeas, kkterror)
    push!(plt, 2, iter, primviol); 
    push!(plt, 3, iter, dualviol); 
    push!(plt, 4, iter, primfeas);
    push!(plt, 5, iter, kkterror); 
    push!(plt, 1, iter, dist); 
    gui()
end

function getDataFilename(prefix::String, case::String, algo::String,
                            num_partitions::Int, perturbation::Number, jacobi::Bool)
    return prefix * 
            basename(case) * 
            "_" * algo * 
            "_n" * string(num_partitions) * 
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

function main_plot(fileprefix::String, savefileprefix::String, case::String, algo::String, jacobi::Bool;
                    partitions, perturbation)
    pyplot()
    p = []
    for i in 1:length(partitions)
        for j in 1:length(perturbation)
            datafile = getDataFilename(fileprefix, case, algo, partitions[i], perturbation[j], jacobi)
            if isfile(datafile)
                savedata = readdlm(datafile)
            else
                savedata = 1e-12*ones(2000, 8)
            end
            push!(p,
                plot(savedata[:,[1,2,3,4]],
                    lab=["distance"  "primviol"  "dualviol" "prinfeas"],
                    lw = 1.8,
                    framestyle = :box,
                    linealpha = 0.7,
                    ylim = [1e-5, 1e1],
                    yscale = :log10))
            if j == 1
                plot!(p[end],
                    guidefontsize = 14,
                    ylabel="parts="* string(partitions[i]))
            end
            if i == 1
                plot!(p[end],
                    titlefontsize = 14,
                    title="perturb="* string(perturbation[j]))
            end
        end
    end
    if !isempty(p)
        plot(p..., layout = (5,5), size = (1000,1000))
        savefig(getSaveFilename(savefileprefix, case, algo, jacobi))
    end
end

function all_plot(fileprefix::String, savefileprefix::String)
    for case in ["case30", "case57", "case118", "case300"]
        for algo in ["aladin", "proxALM"]
            for jacobi in [true]
                if algo == "aladin" && !jacobi
                    continue
                end
                main_plot(fileprefix, savefileprefix, case, algo, jacobi;
                            partitions = [1,2,3,5,10], perturbation=[0, 0.01, 0.1, 0.2, 0.5])
            end
        end
    end
end



