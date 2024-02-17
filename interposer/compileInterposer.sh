#!/bin/bash
rm -rf *.so
#creates the guardian library
/opt/rh/devtoolset-9/root/usr/bin/g++ -O3 -DMODIFIED_PTX -I${CUDA_HOME}/include -fPIC -shared -o guardian_lib.so interposer.cpp -ldl -L${CUDA_HOME}/lib64 -lcudart -lcuda

#creates a library that intercepts cudaLaunchKernel and then calls cuLaunchKernel with the
#same pointer (no ptx). 
/opt/rh/devtoolset-9/root/usr/bin/g++ -O3 -I${CUDA_HOME}/include -fPIC -shared -o nat_lib.so interposer.cpp -ldl -L${CUDA_HOME}/lib64 -lcudart -lcuda

