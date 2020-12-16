```@meta
CurrentModule = ProxAL
```

# Developer reference

## NLP Blocks

The [Formulation](@ref) is decomposed into smaller nonlinear programming (NLP) blocks. Internally, each block is represented as follows.

```@docs
OPFBlockData
opf_block_model_initialize
opf_block_set_objective
opf_block_get_auglag_penalty_expr
opf_block_solve_model
```
