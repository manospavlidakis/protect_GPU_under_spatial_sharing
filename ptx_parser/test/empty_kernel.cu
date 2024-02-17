#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream> 
__device__ int a = 0;
extern "C" __global__ void Mykernel(int *A, int i)
{
	int tid = threadIdx.x;
	if (tid > 5)
		A[i] = tid;
}
