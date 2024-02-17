#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream> 
//__device__  int value2 = 2;
extern "C" __global__ void generateInstruction(unsigned int *A) {
	  unsigned int value = 2;
	  int id = threadIdx.x;
	  A[id + 4] = value;
}
