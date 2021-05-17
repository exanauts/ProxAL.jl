# Usage
`ProxAL.jl` can be called from existing Julia code or REPL, or also from the terminal.

!!! note
    To do: Update documentation to show how to use MPI.

## Julia code or REPL
Install `ProxAL.jl` via the Julia package manager (type `]`):
```julia-repl
pkg> add git@github.com:exanauts/ProxAL.jl.git
pkg> test ProxAL
```
Next, set up and solve the problem as follows. Note that all case files are stored in the `data/` subdirectory. For a full list of model and algorithmic options, see [Model parameters](@ref) and [Algorithm parameters](@ref).
```julia
using ProxAL
using JuMP, Ipopt

# Model/formulation settings
modelinfo = ModelParams()
modelinfo.case_name = "case9"
modelinfo.num_time_periods = 2
modelinfo.num_ctgs = 1
modelinfo.weight_quadratic_penalty_time = 0.1
modelinfo.weight_freq_ctrl = 0.1
modelinfo.time_link_constr_type = :penalty
modelinfo.ctgs_link_constr_type = :frequency_ctrl

# Load case
case_file = "data/" * modelinfo.case_name
load_file = "data/mp_demand/" * modelinfo.case_name * "_oneweek_168"
rawdata = RawData(case_file, load_file)
opfdata = opf_loaddata(rawdata;
                       time_horizon_start = 1,
                       time_horizon_end = modelinfo.num_time_periods,
                       load_scale = modelinfo.load_scale,
                       ramp_scale = modelinfo.ramp_scale)

# Algorithm settings
algparams = AlgParams()
algparams.parallel = false
algparams.decompCtgs = false
algparams.verbose = 0
algparams.optimizer = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
maxρ = 0.1
set_rho!(algparams;
         ngen = length(opfdata.generators),
         modelinfo = modelinfo,
         maxρ_t = maxρ,
         maxρ_c = maxρ)
algparams.mode = :coldstart

if algparams.mode ∈ [:nondecomposed, :lyapunov_bound]
    solve_fullmodel(opfdata, rawdata, modelinfo, algparams)
elseif algparams.mode == :coldstart
    run_proxALM(opfdata, rawdata, modelinfo, algparams)
end
```


## Terminal
The `examples/` directory provides an example of how `ProxAL.jl` can be set up to be used from the terminal. Enter `julia examples/main.jl --help` to get a help message:
```
usage: main.jl [--T T] [--Ctgs CTGS] [--time_unit UNIT]
               [--ramp_value RVAL] [--decompCtgs] [--ramp_constr RCON]
               [--Ctgs_constr CCON] [--load_scale LSCALE]
               [--quad_penalty QPEN] [--auglag_rho RHO]
               [--compute_mode MODE] [-h] case

positional arguments:
  case                 Case name [case9, case30, case118,
                       case1354pegase, case2383wp, case9241pegase]

optional arguments:
  --T T                No. of time periods (type: Int64, default: 10)
  --Ctgs CTGS          No. of line ctgs (type: Int64, default: 0)
  --time_unit UNIT     Select: [hour, minute] (default: "minute")
  --ramp_value RVAL    Ramp value: % Pg_max/time_unit (type: Float64,
                       default: 0.5)
  --decompCtgs         Decompose contingencies
  --ramp_constr RCON   Select: [penalty, equality, inequality]
                       (default: "penalty")
  --Ctgs_constr CCON   Select: [frequency_ctrl, preventive_penalty,
                       preventive_equality, corrective_penalty,
                       corrective_equality, corrective_inequality]
                       (default: "preventive_equality")
  --load_scale LSCALE  Load multiplier (type: Float64, default: 1.0)
  --quad_penalty QPEN  Qaudratic penalty parameter (type: Float64,
                       default: 1000.0)
  --auglag_rho RHO     Aug Lag parameter (type: Float64, default: 1.0)
  --compute_mode MODE  Choose from: [nondecomposed, coldstart,
                       lyapunov_bound] (default: "coldstart")
  -h, --help           show this help message and exit
```

A typical call might look as follows:
```
julia examples/main.jl case9 --T=2 --Ctgs=1 --time_unit=hour --ramp_value=0.5 --load_scale=1.0 --ramp_constr=penalty --Ctgs_constr=frequency_ctrl --auglag_rho=0.1 --quad_penalty=0.1 --compute_mode=coldstart
```


