#!/bin/bash
mkdir -p build
cd build
#export CMAKE_INSTALL_PREFIX="/pfa"
cmake -DCMAKE_INSTALL_PREFIX=/pfa /pfa/c++/PFASamplers
make
make install