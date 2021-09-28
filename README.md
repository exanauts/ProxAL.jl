
# ProxAL.jl

[![][docs-latest-img]][docs-latest-url] ![Run tests](https://github.com/exanauts/ProxAL.jl/workflows/Run%20tests/badge.svg?branch=master)

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://exanauts.github.io/ProxAL.jl/

This is a Julia implementation of a parallel <ins>Prox</ins>imal <ins>A</ins>ugmented <ins>L</ins>agrangian solver for solving multiperiod contingency-constrained ACOPF problems. Please refer to the [documentation][docs-latest-url] for package installation, usage and solver options.

# Installation

The package is under heavy development and relies on non registered Julia packages and versions. This requires to install packages via

```bash
julia --project deps/deps.jl
```

This will also download the case files for TAMU 2K

# Running Example

Executing with [ExaTron](https://github.com/exanauts/ExaTron.jl/) as a backend, `T=2` time periods, and `K=1` contingencies. The example per default uses `ProxAL.CUDADevice`.

```bash
julia --project examples/exatron.jl 2 1
```

## Funding
This research was supported by the Exascale Computing Project (17-SC-20-SC), a joint project of the U.S. Department of Energy’s Office of Science and National Nuclear Security Administration, responsible for delivering a capable exascale ecosystem, including software, applications, and hardware technology, to support the nation’s exascale computing imperative.
