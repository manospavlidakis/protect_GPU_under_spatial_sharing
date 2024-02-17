#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream> 
//__device__  int value2 = 2;
extern "C" __global__ void Mykernel(unsigned int *A, int i, int j)
{
	unsigned int value = 2;
	A = A + i;
	A[value] = value;
}
