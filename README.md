
# ProxAL.jl
---
This is a Julia implementation of a parallel <ins>Prox</ins>imal <ins>A</ins>ugmented <ins>L</ins>agrangian solver for solving multiperiod contingency-constrained ACOPF problems.

## Formulation
The package is designed to solve ACOPF formulations over multiple time periods. The different time periods may have different active and reactive demands, and are linked together via active power ramping constraints: 
```math
<p align="center"><img src="/tex/9ac0b6eb3c993e1889a33ddefbaab6de.svg?invert_in_darkmode&sanitize=true" align=middle width=360.36845295pt height=20.50407645pt/></p>
```
Here, <img src="/tex/b7f7e41df22c4f37982c13f9954f1269.svg?invert_in_darkmode&sanitize=true" align=middle width=20.06232689999999pt height=26.76175259999998pt/> denotes the 'base-case' active power generation level of generator <img src="/tex/934f5567293e2a26bf35336e0fd652dd.svg?invert_in_darkmode&sanitize=true" align=middle width=41.44613879999999pt height=22.465723500000017pt/> in time period <img src="/tex/2b2595d381c04d836f219b7837ded4c2.svg?invert_in_darkmode&sanitize=true" align=middle width=37.916549549999985pt height=22.465723500000017pt/>, and <img src="/tex/4364b58caba8ae1923651f1ca93c1515.svg?invert_in_darkmode&sanitize=true" align=middle width=14.24229014999999pt height=14.15524440000002pt/> denotes its ramping capacity (per unit of time in which <img src="/tex/2f118ee06d05f3c2d98361d9c30e38ce.svg?invert_in_darkmode&sanitize=true" align=middle width=11.889314249999991pt height=22.465723500000017pt/> is defined).

Each single-period ACOPF problem may itself be constrained further by a set of transmission line contingencies, denoted by <img src="/tex/d6328eaebbcd5c358f426dbea4bdbf70.svg?invert_in_darkmode&sanitize=true" align=middle width=15.13700594999999pt height=22.465723500000017pt/>. The active and reactive power generations, and bus voltages must satisfy the following constraints in each time period and each contingency: (i) the power flow equations, (ii) bounds on active and reactive generation and voltage magnitudes, and (iii) line power flow limits. The package allows constraint infeasibility (except variable bounds) by penalizing them in the objective function.

The contingencies in each time period are linked together via their active power generations in one of several ways:
* Preventive mode: active power generation in contingency <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/> must be equal to the base case value.
<p align="center"><img src="/tex/e66380417fa5ee8b54371ab112c1c7a2.svg?invert_in_darkmode&sanitize=true" align=middle width=274.1100219pt height=21.07942155pt/></p>
* Corrective mode: active power generation is allowed to deviate from base case.
```math
<p align="center"><img src="/tex/6d11b26a7afd8c07222c68c1aaccce06.svg?invert_in_darkmode&sanitize=true" align=middle width=428.43874095pt height=21.07942155pt/></p>
```
* Frequency control mode: <img src="/tex/8f5f9043f2a55b3cbaaf5b1b8f638daf.svg?invert_in_darkmode&sanitize=true" align=middle width=22.463953049999994pt height=14.15524440000002pt/> is the (deviation from nominal) system frequency in contingency <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/> of time period <img src="/tex/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode&sanitize=true" align=middle width=5.936097749999991pt height=20.221802699999984pt/>, and <img src="/tex/f91ee1a747e7ea280b801a19c170e6ba.svg?invert_in_darkmode&sanitize=true" align=middle width=17.34161714999999pt height=14.15524440000002pt/> is the droop control parameter of generator <img src="/tex/3cf4fbd05970446973fc3d9fa3fe3c41.svg?invert_in_darkmode&sanitize=true" align=middle width=8.430376349999989pt height=14.15524440000002pt/>. Note that <img src="/tex/8f5f9043f2a55b3cbaaf5b1b8f638daf.svg?invert_in_darkmode&sanitize=true" align=middle width=22.463953049999994pt height=14.15524440000002pt/> are additional decision variables in this case.
```math
<p align="center"><img src="/tex/ce3abef00d04e8cd329f0b9959cfbccc.svg?invert_in_darkmode&sanitize=true" align=middle width=335.6504976pt height=21.07942155pt/></p>
```

#### Overview of solution procedure
The model is decomposed into smaller optimization blocks. The pacakge supports decomposition into (A) single-period multiple-contingency ACOPF problems, and (B) single-period single-contingency ACOPF problems.

This decomposition is achieved by formulating an Augmented Lagrangian with respect to the coupling constraints: in decomposition mode (A), these are the ramping constraints; and in mode (B), these are the ramping as well as contingency-linking constraints.

The formulation is then solved using an iterative ADMM-like Jacobi scheme with a particular choice of proximal weights, by updating first the primal variables (e.g., power generations and voltages) and then the dual variables of the coupling constraints. The Jacobi nature of the update implies that the single-block optimization problems can be solved in parallel. The package allows for the parallel solution of these problems using Julia's `Distributed` computing package.


## Usage
The package can be used from the terminal or from within an existing Julia code or REPL.

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


> **NOTE:** The `--ramp_constr=equality` and `--ramp_constr=penalty` options convert the inequality form of the ramping constraints (see above) into equality constraints by introducing additional slack variables <img src="/tex/25f01eccded518e2f18d05039e57f3f7.svg?invert_in_darkmode&sanitize=true" align=middle width=94.83508319999999pt height=21.18721440000001pt/>.
>
> * In the case of `--ramp_constr=equality`, the ramping constraints are then converted to: <img src="/tex/91abc585ae3ce9740aeceeeeb8ba9396.svg?invert_in_darkmode&sanitize=true" align=middle width=159.16046684999998pt height=26.76175259999998pt/>.
> * In the case of `--ramp_constr=penalty`, further additional (unconstrained) variables <img src="/tex/0e5c9dbcc0707c5ae0bd463348369463.svg?invert_in_darkmode&sanitize=true" align=middle width=19.43641424999999pt height=14.15524440000002pt/> are introduced, and the ramping constraints are converted to: <img src="/tex/f2622e200131ed7c9efc468d3eba64b1.svg?invert_in_darkmode&sanitize=true" align=middle width=199.50994634999998pt height=26.76175259999998pt/>. An additional term QPEN<img src="/tex/9a7982001a9ae424e0990bfc433ca437.svg?invert_in_darkmode&sanitize=true" align=middle width=35.92480814999999pt height=26.76175259999998pt/> is added to the objective function, where QPEN is set using the `--quad_penalty=...` option to the program.
>
> An analogous comment applies to the `--Ctgs_constr={corrective_equality, corrective_penalty, preventive_penalty}` options.

A typical call might look as follows:
```julia
julia src/main.jl case9 --T=2 --Ctgs=1 --time_unit=hour --ramp_value=0.5 --load_scale=1.0 --ramp_constr=penalty --Ctgs_constr=frequency_ctrl --auglag_rho=0.1 --quad_penalty=0.1 --compute_mode=coldstart
```
All case files are stored in the `data/` subdirectory and pulled from there. To use multiple processes (say 2 processes), modify the `julia` call to `julia -p 2`.

### Julia REPL
The package can also be called from existing Julia code. An example follows.
```julia

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
case_file = joinpath("data/case9", modelinfo.case_name)
load_file = joinpath("data/mp_demand/", modelinfo.case_name * "case9_oneweek_168")
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
