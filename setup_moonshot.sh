#!/bin/bash
module load intel mpich hdf5
julia --project -e 'ENV["JULIA_HDF5_PATH"]="/nfs/gce/software/custom/linux-ubuntu18.04-x86_64/hdf5/1.12.1-mpich-3.4.2-intel-parallel-fortran"; ENV["JULIA_MPI_BINARY"]="system"; ENV["JULIA_MPI_PATH"]="/nfs/gce/software/custom/linux-ubuntu18.04-x86_64/mpich/3.4.2-intel"; using Pkg; Pkg.build("MPI"; verbose=true); Pkg.build("HDF5"; verbose=true)'
julia --project -e 'using MPI ; MPI.install_mpiexecjl(force=true ;destdir=".")'
export OMP_NUM_THREADS=1
