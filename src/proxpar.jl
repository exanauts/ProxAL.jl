##
## sctest.jl
using Distributed
@everywhere begin
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()
    using SharedArrays
    using Printf
    using LinearAlgebra
    using JuMP, Ipopt
    using Plots, Measures, DelimitedFiles

    include("./src/opfdata.jl")
    include("./src/params.jl")
    include("./src/mpsolution.jl")
    include("./src/scacopf_model.jl")
    include("./src/mpproxALM.jl")
    include("./src/analysis.jl")
    include("./src/rolling_horizon.jl")
end


##
## x-update step
soltimes = SharedVector{Float64}(T)
@sync for t = 1:T
    function xstep(v)
        opfmodel = opf_model(opfdata, rawdata, t; options = options)
        opf_model_set_objective(opfmodel, opfdata, t; options = options, params = params, primal = x, dual = λ)
        v[:,t] = opf_solve_model(opfmodel)
    end
    @async @spawn begin
        t0 = time()
        xstep(base.colValue)
        soltimes[t] = time() - t0
    end

    #
    # Gauss-Siedel --> immediately update
    #
    if !params.jacobi && !options.two_block
        updatePrimalSolution(x, base.colValue[:,t], base.colIndex, t; options = options)
    end
end