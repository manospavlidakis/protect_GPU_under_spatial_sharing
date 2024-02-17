#include <iostream>
#include <stdio.h>
#include <string>
#include <iostream>
#include <cstring>
#include <fstream>
#include <streambuf>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include <vector>
#include <chrono>
#include <string>
#define TBLOCKS 1
#define THREADS 64
#define RED "\033[1;31m"
#define RESET "\033[0m"

#define CHK(X) if ((err = X) != CUDA_SUCCESS) printf("CUDA error %d at %d\n", (int)err, __LINE__) 

#define CUDA_ERROR_FATAL(err)                                                  \
  cudaErrorCheckFatal(err, __func__, __FILE__, __LINE__)

static void __attribute__((unused))
cudaErrorCheckFatal(CUresult err, const char *func, const char *file,
                    size_t line) {
  const char* err_str = nullptr;
  if (err != CUDA_SUCCESS) {
    cuGetErrorString(err, &err_str);
    std::cerr << RED << func << " error : " << RESET << err_str << std::endl;
    std::cerr << "\t" << file << RED << " Failed at " << RESET << line
              << std::endl;
    exit(1);
  }
} 
std::chrono::high_resolution_clock::time_point s_compute;
std::chrono::high_resolution_clock::time_point e_compute;
// Variables 
CUdevice cuDevice; 
CUcontext cuContext; 
CUmodule cuModule; 
CUfunction ptx_simple; 
CUresult err; 
CUdeviceptr d_A; 
CUdeviceptr d_B;
CUdeviceptr d_C;

void ConstantInit(int *data, int size, int val) {
    for (int i = 0; i < size; ++i)
        data[i] = val;
}
int main(int argc, char **argv)
{
	if (argc < 2) {
		std::cerr << "Usage: Add the ptx file." << std::endl;
		return 1;
	}
	std::string ptx_file_name = argv[1];
	std::cerr<<"PTX: "<< ptx_file_name<<std::endl;

	int devID = 0;
	size_t dim = TBLOCKS*THREADS;
	CUDA_ERROR_FATAL(cuInit(0));
	CUDA_ERROR_FATAL(cuDeviceGet(&cuDevice, devID));
	CUDA_ERROR_FATAL(cuCtxCreate(&cuContext, 0, cuDevice));
	std::ifstream my_file(ptx_file_name);
	std::string my_ptx((std::istreambuf_iterator<char>(my_file)), std::istreambuf_iterator<char>());
	// Create module from PTX
	CUDA_ERROR_FATAL(cuModuleLoadData(&cuModule, my_ptx.c_str()));
	// Get function handle from module
	CUDA_ERROR_FATAL(cuModuleGetFunction(&ptx_simple, cuModule, "Mykernel"));
	int *h_A = (int *) malloc(dim*sizeof(int));
	int *h_B = (int *) malloc(dim*sizeof(int));
  
	CUDA_ERROR_FATAL(cuMemAlloc(&d_A, dim*sizeof(int)));

	ConstantInit(h_A, dim, 0);
	ConstantInit(h_B, dim, 8);

	CUDA_ERROR_FATAL(cuMemcpyHtoD(d_A, h_A, dim*sizeof(int)));
	int param = 19;
	int index = 1;
	void *args[] = { &d_A, &index, &param };
	s_compute = std::chrono::high_resolution_clock::now();
	CUDA_ERROR_FATAL(cuLaunchKernel(ptx_simple, TBLOCKS, 1, 1,
				THREADS, 1, 1, 0, NULL, args, NULL));

	cuCtxSynchronize();

	e_compute = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> compute_milli =
		e_compute - s_compute;
	std::cerr << "Computation: " << compute_milli.count() << " ms" << std::endl;

	CUDA_ERROR_FATAL(cuMemcpyDtoH(h_B, d_A, dim*sizeof(int)));
		
	std::cerr<<"h_B[ "<< index <<" ]: "<<h_B[index]<<std::endl;

	std::cerr<<"Done with simple kernel"<<std::endl;
	return 0;
}
