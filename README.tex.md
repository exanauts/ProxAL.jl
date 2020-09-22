
# ProxAL.jl

This is a Julia implementation of a parallel <ins>Prox</ins>imal <ins>A</ins>ugmented <ins>L</ins>agrangian solver for solving multiperiod contingency-constrained ACOPF problems.

## Formulation
The package is designed to solve ACOPF formulations over multiple time periods. The different time periods may have different active and reactive demands, and are linked together via active power ramping constraints: 
$$
-r_g \leq p^0_{g,t-1} - p^0_{g,t} \leq r_g \qquad \forall g \in G, \; \forall t \in T \setminus \{1\}.
$$

Here, $p^0_{gt}$ denotes the 'base-case' active power generation level of generator $g \in G$ in time period $t \in T$, and $r_g$ denotes its ramping capacity (per unit of time in which $T$ is defined).

Each single-period ACOPF problem may itself be constrained further by a set of transmission line contingencies, denoted by $K$. The active and reactive power generations, and bus voltages must satisfy the following constraints in each time period and each contingency: (i) the power flow equations, (ii) bounds on active and reactive generation and voltage magnitudes, and (iii) line power flow limits. The package allows constraint infeasibility (except variable bounds) by penalizing them in the objective function.

The contingencies in each time period are linked together via their active power generations in one of several ways:
* Preventive mode: active power generation in contingency $k$ must be equal to the base case value.
$$
p_{gt}^k = p_{gt}^0 \qquad \forall g \in G, \; \forall k \in K, \; \forall t \in T.
$$

* Corrective mode: active power generation is allowed to deviate from base case.
$$
0.1\times r_g \leq p_{gt}^k - p_{gt}^0 \leq 0.1 \times r_g \qquad \forall g \in G, \; \forall k \in K, \; \forall t \in T.
$$

* Frequency control mode: $\omega_{kt}$ is the (deviation from nominal) system frequency in contingency $k$ of time period $t$, and $\alpha_g$ is the droop control parameter of generator $g$. Note that $\omega_{kt}$ are additional decision variables in this case.
$$
p_{gt}^k = p_{gt}^0 + \alpha_g \omega_{kt} \qquad \forall g \in G, \; \forall k \in K, \; \forall t \in T.
$$

#### Algorithm
The model is decomposed into smaller optimization blocks. The pacakge supports decomposition into (A) single-period multiple-contingency ACOPF problems, and (B) single-period single-contingency ACOPF problems.

This decomposition is achieved by formulating an Augmented Lagrangian with respect to the coupling constraints: in decomposition mode (A), these are the ramping constraints; and in mode (B), these are the ramping as well as contingency-linking constraints.

The decomposed formulation is solved using an iterative ADMM-like Jacobi scheme with a particular choice of proximal weights, by updating first the primal variables (e.g., power generations and voltages) and then the dual variables of the coupling constraints. The Jacobi nature of the update implies that the single-block optimization problems can be solved in parallel. The package allows for the parallel solution of these problems using Julia's `Distributed` computing package.


## Usage
The package can be used from the terminal or from within an existing Julia code or REPL.

### Julia REPL
Install `ProxAL` via the Julia package manager (type `]`):
```julia
pkg> add git@github.com:exanauts/ProxAL.jl.git
pkg> test ProxAL
```
Next, set up and solve the problem as follows. Note that all case files are stored in the `data/` subdirectory. For a full list of model and algorithmic options, see `src/proxALMutil.jl`.
```julia
using ProxAL

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
maxρ = 0.1
set_rho!(algparams;
         ngen = length(opfdata.generators),
         modelinfo = modelinfo,
         maxρ_t = maxρ,
         maxρ_c = maxρ)
algparams.mode = :coldstart

if algparams.mode ∈ [:nondecomposed, :lyapunov_bound]
    solve_fullmodel(opfdata, rawdata; modelinfo = modelinfo, algparams = algparams)
elseif algparams.mode == :coldstart
    run_proxALM(opfdata, rawdata; modelinfo = modelinfo, algparams = algparams)
end
```

### Terminal
Enter `julia src/main.jl --help` to get a help message:
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


> **NOTE:** The `--ramp_constr=equality` and `--ramp_constr=penalty` options convert the inequality form of the ramping constraints (see above) into equality constraints by introducing additional slack variables $0 \leq s_{gt} \leq 2r_g$.
>
> * In the case of `--ramp_constr=equality`, the ramping constraints are then converted to: $p^0_{g,t-1} - p^0_{gt} + s_{gt} = r_g$.
> * In the case of `--ramp_constr=penalty`, further additional (unconstrained) variables $z_{gt}$ are introduced, and the ramping constraints are converted to: $p^0_{g,t-1} - p^0_{gt} + s_{gt} + z_{gt} = r_g$. An additional term QPEN$\cdot\Vert z \rVert^2$ is added to the objective function, where QPEN is set using the `--quad_penalty=...` option to the program.
>
> An analogous comment applies to the `--Ctgs_constr={corrective_equality, corrective_penalty, preventive_penalty}` options.

A typical call might look as follows:
```julia
julia src/main.jl case9 --T=2 --Ctgs=1 --time_unit=hour --ramp_value=0.5 --load_scale=1.0 --ramp_constr=penalty --Ctgs_constr=frequency_ctrl --auglag_rho=0.1 --quad_penalty=0.1 --compute_mode=coldstart
```
To use multiple processes (say 2 processes), modify the `julia` call to `julia -p 2`.

