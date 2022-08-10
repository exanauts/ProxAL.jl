#!/bin/bash
export JULIA_HDF5_PATH="/disk/hdf5/hdf5-1.12.2/build/bin"
export JULIA_MPI_BINARY="system"
export OMP_NUM_THREADS=1
julia --project -e 'using Pkg; Pkg.build("MPI"; verbose=true); Pkg.build("HDF5"; verbose=true)'
julia --project -e 'using MPI ; MPI.install_mpiexecjl(force=true ;destdir=".")'
