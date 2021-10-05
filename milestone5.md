# ECP ExaSGD Milepost 5
This documents describes the completion of the project ExaSGD milepost 5 as stated by:

> Run formulation 4 using full ExaSGD software stack on Tulip and/or Summit. Use 1,000 bus grid model with 5+ scenarios and 100-500 contingencies and 2 periods. All computations inside optimization loops run on GPU.

## Preliminary notes
"Tulip" was an early access AMD system provided by Cray. In the meantime, we moved our code base to the superseding AMD system "Spock" at OLCF. The milepost was completed on "Spock" and "Summit", while only executing a small job on Spock to demonstrate our portability strategy.

## Prerequisites

* Julia 1.6 or 1.7 [https://julialang.org/](https://julialang.org/)
* CUDA (Summit), or ROCm 4.2 (Spock)


## General Setup
The Julia depot path should not be in the home directory for latency reasons. Since Summit and Spock share the same filesystems, but are different CPU architectures, make sure to load the correct depot path.

```bash
export JULIA_DEPOT_PATH=/ccs/proj/csc359/$USER/julia_depot
```

The exact package versions are defined by manifest files
* `Manifest_Summit.toml` for Summit,
* and `Manifest_Spock.toml` for Spock.

The environments will be instantiated later based on these manifests.

The data is from our `ExaData` repository and is cloned with

```bash
git clone git@github.com:exanauts/ExaData.git
```

into the `ProxAL` folder that you will checkout later.

## Summit

Disable the download of ROCm binaries through the Julia artifacts infrastructure.

```bash
export JULIA_AMDGPU_DISABLE_ARTIFACTS=1
```

Checkout ProxAL and switch to the develop branch.

```bash
git clone git@github.com:exanauts/ProxAL.jl.git
git checkout ecp/milepost5
```

In the ProxAL folder instantiate the environment as defined by the Manifest.toml file.

```bash
julia --project
(ProxAL) pkg> instantiate
```

Verify that CUDA.jl can be successfully loaded

```bash
julia> using CUDA
julia> x = ones(10) |> CuVector
julia> x .= 2.0
```

Before job submission, compile the Julia MPI package `MPI.jl` with the system MPI library via the following command.

```bash
julia --project -e 'ENV["JULIA_MPI_BINARY"]="system"; using Pkg; Pkg.build("MPI"; verbose=true)'
```

Our job submission script for the TAMU2k case with `K=600` contingencies and `T=6` periods is

```
#!/bin/bash
# Begin LSF Directives
#BSUB -P CSC359
#BSUB -W 1:00
#BSUB -nnodes 60
#BSUB -J proxal-tests

cd $MEMBERWORK/csc359/michel/summit/git/ProxAL.jl
date
module load cuda
RANKS=360
export JULIA_CUDA_VERBOSE=1
export JULIA_DEPOT_PATH=$PROJWORK/csc359/michel/julia_depot
export JULIA_MPI_BINARY=system
export JULIA_EXEC=/ccs/proj/csc359/michel/julia-1.7.0-rc1/bin/julia
jsrun -n $RANKS -r 6 -c 1 -a 1 -g 1 julia --color=no --project ./examples/exatron.jl 6 600
```

This executes the Julia code in `examples/exatron.jl` which loads the TAMU 2000 case file per default.

## Spock

### Setup

Disable the download of ROCm binaries through the Julia artifacts infrastructure.

```bash
export JULIA_AMDGPU_DISABLE_ARTIFACTS=1
```


Checkout ProxAL and switch to the `ecp/milepost5` branch.

```bash
git clone git@github.com:exanauts/ProxAL.jl.git
git checkout ecp/milepost5-spock
```

Our runs were done using ROCm 4.2.

```bash
module load ROCm/4.2
```

In the ProxAL folder instantiate the environment as defined in the `deps/deps.jl` file.

```bash
julia --project deps/deps.jl
```

Verify that AMDGPU.jl can be successfully loaded

```bash
julia> using AMDGPU
julia> x = ones(10) |> ROCVector
julia> x .= 2.0
```

### Execution

julia --project examples/exatron.jl 2

This loads a IEEE 9-bus case and runs the problem with `T=2` periods. The version of [ExaTron](https://github.com/exanauts/ExaTron.jl/) used as a backend here relies on [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) to run both on CUDA and ROCm architectures.



