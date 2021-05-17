```@meta
CurrentModule = ProxAL
```

# Developer reference

## NLP Blocks

The [Formulation](@ref) is decomposed into smaller nonlinear programming (NLP) blocks.
Blocks are coupled together using a `OPFBlocks` structure.
```@docs
OPFBlocks
```

Internally, each block is represented as follows.

```@docs
AbstractBlockModel
JuMPBlockModel
ExaBlockModel
init!
optimize!
set_objective!
get_solution
add_variables!

```
