```@meta
CurrentModule = ProxAL
```

# NLP blocks and backends

## NLP Blocks

The [Formulation](@ref) is decomposed into smaller nonlinear programming (NLP) blocks.
Blocks are coupled together using a `OPFBlocks` structure.
```@docs
OPFBlocks
```

Internally, each block is represented as follows.

```@docs
AbstractBlockModel
init!
add_variables!
set_objective!
optimize!
set_start_values!
get_solution

```

## Backends

```@docs
JuMPBackend
JuMPBlockBackend
ExaTronBackend
TronBlockBackend
ExaPFBackend
ExaBlockBackend

```

## OPF

```@docs
opf_block_get_auglag_penalty_expr
```
