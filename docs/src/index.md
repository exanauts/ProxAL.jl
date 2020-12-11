# ProxAL

[ProxAL.jl](https://github.com/exanauts/ProxAL.jl) is a Julia package to solve multiperiod contingency-constrained AC Optimal Power Flow (ACOPF) problems. Its main feature is a distributed parallel implementation, which allows running on high-performance computing architectures. 

This document describes the algorithm, API and main functions of `ProxAL.jl`.

## Table of contents

### Manual

```@contents
Pages = [
    "man/formulation.md",
    "man/algorithm.md",
    "man/usage.md",
]
Depth = 1
```

### Library

```@contents
Pages = [
    "lib/modelparams.md",
    "lib/algparams.md",
    "lib/algorithm.md",
    "lib/developer.md",
]
Depth = 1
```

## Funding

This research was supported by the [Exascale Computing Project](https://www.exascaleproject.org/), a joint project of the U.S. Department of Energy’s Office of Science and National Nuclear Security Administration.
