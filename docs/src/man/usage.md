# Usage
`ProxAL.jl` can be called from existing Julia code or REPL. The package is under heavy development and relies on non-registered Julia packages and versions. This requires to install packages via:
```shell
$ git clone https://github.com/exanauts/ProxAL.jl.git
$ cd ProxAL.jl
$ julia --project "using Pkg ; Pkg.instantiate()"
```

## Example
We can set up and solve a problem as follows. For a full list of model and algorithmic options, see [Model parameters](@ref) and [Algorithm parameters](@ref).

Consider the following `example.jl` using the `JuMP` backend, `Ipopt` solver, and using `MPI`:
```julia
using ProxAL
using JuMP, Ipopt
using MPI
using LazyArtifacts

MPI.Init()

# Model/formulation settings
modelinfo = ModelParams()
modelinfo.num_time_periods = 10
modelinfo.num_ctgs = 0
modelinfo.allow_line_limits = false

# Load case in MATPOWER format
# This automatically loads data from https://github.com/exanauts/ExaData
# You may also provide your own case data
case_file = joinpath(artifact"ExaData", "ExaData", "case118.m")
load_file = joinpath(artifact"ExaData", "ExaData", "mp_demand", "case118_oneweek_168")

# Choose the backend
backend = ProxAL.JuMPBackend()

# Algorithm settings
algparams = AlgParams()
algparams.verbose = 1
algparams.optimizer = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
algparams.tol = 1e-3 # tolerance for convergence

# Solve the problem
nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, backend, MPI.COMM_WORLD)
runinfo = ProxAL.optimize!(nlp)

@show(runinfo.iter)                  # number of iterations
@show(runinfo.maxviol_t_actual[end]) # ramping violation at last iteration
@show(runinfo.maxviol_d[end])        # dual residual at last iteration

MPI.Finalize()
```

To execute this file with `2` MPI processes:
```shell
$ mpiexec -n 2 julia --project example.jl
```

To disable MPI, simply pass `nothing` as the last argument to `ProxALEvaluator` (or omit the argument entirely) and you can simply run:
```shell
$ julia --project example.jl
```

An example using the `ExaTron` backend with `ProxAL.CUDABackend` (GPU) can be found in `examples/exatron.jl`.
