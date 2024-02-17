# Interposer (guardian library)
1. compile using the compileInterposer.sh. This script create two libraries: guardian and native. Guardian lib performs all checks while native just intercepts cudaLaunchKernel and call cuLaunchKernel with the same pointer.
2. run: export CUDA_FORCE_PTX_JIT=1; LD_PRELOAD=./guardian_lib.so ./a.out

* SOS if you do not add export CUDA_FORCE_PTX_JIT=1 then it uses ampere_sgemm_128x128_nn from cubin

# Compile an application that uses a CUDA-accelerated library
1. nvcc simple_sgem.cpp -lcublas_static -lcublasLt_static -lculibos -lcudart


