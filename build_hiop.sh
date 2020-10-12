#!/bin/bash

git submodule init
git submodule update

mkdir -p build
cd build
cmake -DHIOP_USE_MPI=OFF -DCMAKE_BUILD_TYPE=Debug -DHIOP_BUILD_SHARED=ON -DHIOP_BUILD_STATIC=ON -DCMAKE_INSTALL_PREFIX=$PWD ../hiop
make -j
