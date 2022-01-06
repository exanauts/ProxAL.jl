# Algorithm

The [Formulation](@ref) is decomposed into smaller optimization blocks. Specifically, `ProxAL.jl` supports decomposition into:
1. single-period multiple-contingency ACOPF problems, and 
2. single-period single-contingency ACOPF problems.

This decomposition is achieved by formulating an Augmented Lagrangian with respect to the coupling constraints: in decomposition mode 1, these are the ramping constraints; and in mode 2, these are the ramping as well as contingency-linking constraints.

The decomposed formulation is solved using an iterative ADMM-like Jacobi scheme with proximal terms, by updating first the primal variables (e.g., power generations and voltages) and then the dual variables of the coupling constraints. The Jacobi nature of the update implies that the single-block nonlinear programming (NLP) problems can be solved in parallel. `ProxAL.jl` allows the parallel solution of these NLP block subproblems using the `MPI.jl` package.

The single-block NLP problems can be solved using one of three backends (see [NLP blocks and backends](@ref) for details):
1. [JuMP](https://github.com/jump-dev/JuMP.jl) to solve the ACOPF in full-space (with a user-provided NLP solver)
2. [ExaTron](https://github.com/exanauts/ExaTron.jl) to solve the ACOPF using component-based decomposition
3. [ExaPF](https://github.com/exanauts/ExaPF.jl) to solve the ACOPF in reduced-space

Further details about the algorithm can be found in the preprint.
