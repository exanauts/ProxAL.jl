# Usage
`ProxAL.jl` can be called from existing Julia code or REPL, or also from the terminal.

!!! note
    To do: Update documentation to show how to use `ExaTron`, `ExaPF` backends, as well as other solvers like `MadNLP`.

## Julia code or REPL
Install `ProxAL.jl` via the Julia package manager (type `]`):
```julia-repl
pkg> add git@github.com:exanauts/ProxAL.jl.git
pkg> test ProxAL
```
Next, set up and solve the problem as follows. Note that all case files are stored in the `data/` subdirectory. For a full list of model and algorithmic options, see [Model parameters](@ref) and [Algorithm parameters](@ref).

Consider the following `example.jl` using the `JuMP` backend, `Ipopt` solver, and `MPI`:
```julia
using ProxAL
using JuMP, Ipopt
using MPI

MPI.Init()

# Model/formulation settings
modelinfo = ModelParams()
modelinfo.case_name = "case9"
modelinfo.num_time_periods = 2
modelinfo.num_ctgs = 1
modelinfo.weight_freq_ctrl = 0.1
modelinfo.time_link_constr_type = :penalty
modelinfo.ctgs_link_constr_type = :frequency_ctrl

# Load case in MATPOWER format
case_file = "data/$(modelinfo.case_name).m"
load_file = "data/mp_demand/$(modelinfo.case_name)_oneweek_168"

# Algorithm settings
algparams = AlgParams()
algparams.optimizer = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
algparams.decompCtgs = false
algparams.verbose = 1

nlp = ProxALEvaluator(case_file, load_file, modelinfo, algparams, JuMPBackend())
runinfo = ProxAL.optimize!(nlp)
```

To execute this file with `2` MPI processes, you can call
```
mpiexec -n 2 julia --project=. example.jl
```


