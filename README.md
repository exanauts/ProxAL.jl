
# Decomposition-based solver for multi-period security-constrained ACOPF

Julia package for solving multi-period security-constrained ACOPF problems.

## Brief description of algorithm
The problem is decomposed into single-period single-scenario ACOPF problems by formulating an Augmented Lagrangian (AL) with respect to the cross-period/cross-scenario constraints (e.g., ramping constraints in multi-period ACOPF and AGC constraints in security-constrained ACOPF). The AL formulation is then solved by iteratively updating first the solution vectors in each single-period single-scenario problem using a Jacobi-like update and then the dual variables (as in the standard AL method).

The Jacobi nature of the update implies that the single-period single-scenario problems can be solved in parallel. The package allows for the parallel solution of these problems.

Currently, only multi-period or only security constraints are supported, i.e., the package does not address both simultaneously.


# Usage
To test the multi-period ACOPF on case118 with 50 time periods, run:
```julia
$ julia --project=. src/mptest.jl data/case118 50
```

To test security-constrained ACOPF on case118 with 50 scenarios, run:
```julia
$ julia --project=. src/sctest.jl data/case118 50
```


