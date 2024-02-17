#include <cuda_runtime.h> //Runtime API
#include <cuda.h> //Driver API
#include <cublas_v2.h>
#include <stdio.h>
#include <unistd.h>
#include <dlfcn.h>
#include <string>
#include <iostream>
#include <cstring>
#include <unordered_map>
#include <streambuf>
#include <chrono>
#include "rapidjson/document.h"
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <sstream>
#include <time.h>
#include <cstdint>
//#define DEBUG
#define NO_NEW_PTX
#define PARTITION_NUM 2
#define TIMERS
#define RED "\033[1;31m"
#define RESET "\033[0m"
uint64_t s_compute;
uint64_t e_compute;

uint64_t s_index_map1;
uint64_t e_index_map1;


uint64_t s_args;
uint64_t e_args;


std::unordered_map<const void*, char *> ptr2name;
// key: krnl name, values: ptx, params
std::unordered_map<std::string, std::pair<std::string, int>> kernelMap;
std::unordered_map<std::string, std::pair<std::string, int>> kernelMap86;

// key: ptx, values: krnl name
std::unordered_map<std::string, std::vector<std::string>> PTXMap;
std::unordered_map<std::string, std::vector<std::string>> PTXMap86;

// key: ptx, value cumodule
std::unordered_map<std::string, CUmodule> ptx2module;

// key: Kernel name, value CUfunction
std::unordered_map<std::string, CUfunction> name2CUfunc;

//Map with key: krnl_ptr and values a pair of: krnl_name, ptx, #params
//std::unordered_map<const void*, std::tuple<std::string ,std::string, int>> ptr2ptx_param;

//Map with key: krnl_ptr and values a pair of: CUfunction, #params
std::unordered_map<const void*, std::pair<CUfunction, int>> ptr2CUfunc_param;

std::chrono::high_resolution_clock::time_point s_map_chrono;
std::chrono::high_resolution_clock::time_point e_map_chrono;
/*
std::chrono::high_resolution_clock::time_point s_1;
std::chrono::high_resolution_clock::time_point e_1;

std::chrono::high_resolution_clock::time_point s_2;
std::chrono::high_resolution_clock::time_point e_2;
*/
using namespace std;
int firstcudaLaunch = 0;

uint64_t rdtsc() {
  unsigned int lo, hi;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((uint64_t)hi << 32) | lo;
}

CUresult err;
#define CUDA_ERROR_FATAL(err)                                                  \
  cudaErrorCheckFatal(err, __func__, __FILE__, __LINE__)

static void __attribute__((unused))
cudaErrorCheckFatal(CUresult err, const char *func, const char *file,
                    size_t line) {
  const char* err_str = nullptr;
  if (err != CUDA_SUCCESS) {
    cuGetErrorString(err, &err_str);
    std::cerr << RED << func << " error : " << RESET << err_str<<" "<<err<< std::endl;
    std::cerr << "\t" << file << RED << " Failed at " << RESET << line
              << std::endl;
    exit(1);
  }
}

#define CUDA_ERROR_FATAL_RUNTIME(err)                                                  \
  cudaErrorCheckFatal_Runtime(err, __func__, __FILE__, __LINE__)

static void __attribute__((unused))
cudaErrorCheckFatal_Runtime(cudaError_t err, const char *func, const char *file,
                    size_t line) {
  if (err != cudaSuccess) {
    std::cerr << RED << func << " error : " << RESET << cudaGetErrorString(err)
              << std::endl;
    std::cerr << "\t" << file << RED << " Failed at " << RESET << line
              << std::endl;
    exit(1);
  }
}

#define CHK(X) if ((err = X) != CUDA_SUCCESS) printf("CUDA error %d at %d\n", (int)err, __LINE__) 


typedef struct {
    size_t alloc_size;
    void* base_alloc_ptr;
    void* end_alloc_ptr;

} Allocation;

typedef struct {
    void* partition_ptr;
    size_t partition_size;
    size_t partition_free_space;
    pid_t pid;
    int bits2shift;
    long int mask;
    //std::vector <Allocation> partition_allocations;
    int allocation_cnt = 0;
} Partition;

extern "C"{
int firstRegisterFunc = 0;
std::string read_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error: could not open file " << filename << std::endl;
	abort();
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}
static Partition* partitions = nullptr;
static pid_t prev_pid = 0;
static int prev_partition_index = 0;
static void* prev_end_ptr = NULL;
int partition_index = -1;
/**
 * Alignment is the desired alignment in bytes. This should be a power of two.
 * Ptr is a pointer to the memory address that needs to be aligned.
 *
 * */
inline void *align(std::size_t alignment, void *ptr) noexcept {
    const auto intptr = reinterpret_cast<std::uintptr_t>(ptr);
    const auto aligned = (intptr - 1u + alignment) & -alignment;
    return reinterpret_cast<void *>(aligned);
}
uintptr_t get_mask(uintptr_t addr, uintptr_t alignment) {
    //std::cout << "Original address: 0x" << std::hex << addr << std::endl;	
    uintptr_t aligned_addr = addr & ~(alignment - 1);
    //std::cout << "Aligned address-1: 0x" << std::hex << aligned_addr-1 << std::endl;	
    return aligned_addr - alignment;
}

uint64_t get_partition_size(uint32_t n, uint64_t gpu_memory_size)
{
    // Compute the size of each partition
    const uint64_t partition_size = gpu_memory_size / n;

    // Determine the power of two that is greater or equal to the partition size
    const uint64_t pow2_partition_size = pow(2, floor(log2(partition_size)));
    //const uint64_t pow2_partition_size = 1ull << (64 - __builtin_clzll(partition_size));

    // Create the partitions by setting the appropriate bits
    uint64_t remaining_memory = gpu_memory_size;
    for (uint32_t i = 0; i < n; ++i) {
        const uint64_t partition_size = std::min(pow2_partition_size, remaining_memory);
        remaining_memory -= partition_size;
    }

    // Return the size of each partition
    return pow2_partition_size;
}
int numberOfDigits(int num) {
    if (num == 0) {
        return 1;
    }

    int count = 0;
    if (num < 0) {
        count++;
    }

    while (num != 0) {
        count++;
        num /= 10;
    }

    return count-1;
}
cudaError_t (*cudaMalloc_original)(void **, size_t);
cudaError_t cudaMalloc(void **devPtr, size_t size) {
    size_t partition_size;
#ifdef DEBUG
    std::cerr<<"Intercept Malloc!!"<<std::endl;
#endif
    pid_t pid = getpid();
    //std::cerr<<"PID: "<<pid<<" PREV: "<<prev_pid<<std::endl;
    if (!cudaMalloc_original) {
        cudaMalloc_original = (cudaError_t (*)(void **, size_t))dlsym(RTLD_NEXT, "cudaMalloc");
    }
    if (partitions == nullptr) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        size_t total_memory = deviceProp.totalGlobalMem;
	partition_size = get_partition_size (PARTITION_NUM, total_memory);
#ifdef DEBUG
        std::cerr<<"Total memory: "<< total_memory<<" ,Partition size: "
                <<partition_size<<std::endl;
#endif
        partitions = new Partition[PARTITION_NUM];
	for (int i = 0; i < PARTITION_NUM; i++) {
            partitions[i].partition_size = partition_size;
            partitions[i].pid = -1;
            void * tmp_ptr;
            CUDA_ERROR_FATAL_RUNTIME(cudaMalloc_original(&tmp_ptr, partition_size));
//	    std::cerr<<"i: "<<i<<" allocation size: "<<partition_size
//		    <<" returned ptr: "<<tmp_ptr<<std::endl;
            partitions[i].partition_ptr = align(partition_size, tmp_ptr);
	    uintptr_t highest_ptr = (uintptr_t) partitions[i].partition_ptr
	    	   + partitions[i].partition_size;
	    
	    //uintptr_t mask = get_mask((uintptr_t)partitions[i].partition_ptr, 1);
	    uintptr_t mask = get_mask(highest_ptr, 1);
	    std::cerr << "Highest: " << std::hex << highest_ptr
		    <<" ,Mask: "<< mask << std::dec <<" ,Diff: "
		    << highest_ptr - mask
		    << std::endl;

#ifdef DEBUG
	    if (highest_ptr - mask != 1){
	    	std::cerr<<"mask and partition has diff greater than 1!"<<std::endl;
		abort();
	    }
#endif
            partitions[i].mask = mask;
            partitions[i].partition_free_space = partition_size;
        }
    }
    // Find the next free partition to allocate based on pid
    //partitions[0].pid = 3254;
    for (int i = 0; i < PARTITION_NUM; i++) {
        if (pid != prev_pid){
            //check if partition is not assigned to another process
            if (partitions[i].pid == -1) {
                partition_index = i;
                prev_partition_index = partition_index;
                break;
            }
        }else{
            partition_index = prev_partition_index;
        }
    }
    /*
    if (partition_index == -1) {
        while (is_partition_unassigned()) {
            sleep(1);
        }
        partition_index = pid % PARTITION_NUM;
    }*/
    Partition* partition = &partitions[partition_index];
    partition->pid = pid;
#ifdef DEBUG
    std::cerr << "Selected partition: "<< partition_index <<" ,pid: "
            << partitions[partition_index].pid <<" ,free memory: "
            << partitions[partition_index].partition_free_space
            <<std::endl;
#endif
    if (size < partitions[partition_index].partition_free_space){
        partitions[partition_index].partition_free_space =
                partitions[partition_index].partition_free_space - size;
#ifdef DEBUG
        std::cerr<<"Remaining space left: "
                << partitions[partition_index].partition_free_space<<std::endl;
#endif
        int current_partition_allocations = partitions[partition_index].allocation_cnt;
/*	std::cerr << "Current partition allocations: "
		<< current_partition_allocations <<" partition index: "
		<< partition_index << std::endl;*/
	Allocation allocation;
	// partition_allocations vector is empty!!!!
	//allocation = partition->partition_allocations[current_partition_allocations];
        allocation.alloc_size = size;
	//std::cerr<<"allocation size: "<<allocation.alloc_size<<std::endl;
	//std::cerr<<"previous end ptr: "<<prev_end_ptr<<std::endl;
        // If it is the first allocation 
        if (prev_end_ptr == NULL)
            // Assign the partition ptr
            allocation.base_alloc_ptr = partition->partition_ptr;
        else
            // Assign the end ptr of the (power of two) previous segment
            allocation.base_alloc_ptr = align(256,prev_end_ptr);
	//allocation.base_alloc_ptr = align(256, prev_end_ptr);
        // Now the end ptr is the base + size of allocation
        allocation.end_alloc_ptr
                = (void*)(((unsigned long long) allocation.base_alloc_ptr) + size);
        // store the allocation end ptr to the previous ptr (to be used in next allocations)
        prev_end_ptr = allocation.end_alloc_ptr;
        // increase the segment num of the current parition
        partitions[partition_index].allocation_cnt ++;
	// return the device pointer 
        *devPtr =  allocation.base_alloc_ptr;

        void* highest_addr_partition =
                (void*)(((unsigned long long)partitions[partition_index].partition_ptr)
                              + partitions[partition_index].partition_size);
#ifdef DEBUG
        std::cerr<<"Partition-> Lowest addr: "<< partitions[partition_index].partition_ptr
                <<" , Highest addr: "<< std::hex<< highest_addr_partition
                <<" , Mask: " << partitions[partition_index].mask<<std::dec
                <<" , Size: " << partitions[partition_index].partition_size << std::endl;

        std::cerr<<"Allocation-> Lowest addr (Devive ptr): " << allocation.base_alloc_ptr
                <<" , Size: "<< size
                <<" , Allocations num: "<< partitions[partition_index].allocation_cnt - 1
                <<" , Next allocation will start from: "<< prev_end_ptr
//                <<" , Rounded next allocation will start from: "
//		<< align(128*1024, prev_end_ptr)
                <<" , Mask: " << std::hex << partitions[partition_index].mask<<std::dec
                //<<" , Devive ptr: "<<*devPtr
                <<std::endl;
        if (*devPtr > highest_addr_partition){
                std::cerr<<"Allocation is outside the partition!"<<std::endl;
                abort();
        }

        std::cerr<<"---------------------------------------"<<std::endl;
#endif
	//Store the allocation
	//partition->partition_allocations[current_partition_allocations] = allocation;
        prev_pid = pid;
        return cudaSuccess;

    }else{
        std::cerr<<"There is no space left in the "<<partition_index<<" partition"
                << " for the requested allocation "<< size
                <<" remaining space "<<partitions[partition_index].partition_free_space
                << std::endl;
        return cudaErrorMemoryAllocation;
    }
}
void createModuleMap() {
    CUdevice cuDevice; 
    CUcontext cuContext;
    int devID = 0;
    CUDA_ERROR_FATAL(cuInit(0));
    CUDA_ERROR_FATAL(cuDeviceGet(&cuDevice, devID));
    CUDA_ERROR_FATAL(cuCtxCreate(&cuContext, 0, cuDevice));

    CUmodule cuModule;

    std::cerr<<"Size of kernelMap_param: "<<kernelMap.size()<<std::endl;
    for (const auto& entry : PTXMap) {
        const std::string& ptx = entry.first;
	// Specify additional options for module loading
	std::string path = "./Interposer/caffe_libs/" + ptx;  
	std::ifstream my_file(path);
	std::string my_ptx((std::istreambuf_iterator<char>(my_file)), std::istreambuf_iterator<char>());
        //std::cerr<<"path: "<<path<<std::endl;
	//CUDA_ERROR_FATAL(cuModuleLoadDataEx(&cuModule, my_ptx.c_str(), jitNumOptions, jitOptions, jitOptVals));
	CUDA_ERROR_FATAL(cuModuleLoadData(&cuModule, my_ptx.c_str()));
	ptx2module.emplace(ptx, cuModule);
    }
}

void createFunctionMap() 
{
    CUfunction kernelFunc;
    //std::cerr<<"Size of kernelMap_param: "<<kernelMap.size()<<std::endl;
    for (auto& entry : PTXMap) {
        const std::string& ptxName = entry.first;
	//std::cerr<<"PTX: "<<ptxName<<std::endl;
	CUmodule cuModule = ptx2module.at(ptxName);
	std::vector<std::string>& values = entry.second; // get the vector of values
	for (std::string& krnl : values) {
	    const char* fname = krnl.c_str();
	    //std::cerr<<"Function name: "<<fname<<std::endl;
	    CUDA_ERROR_FATAL(cuModuleGetFunction(&kernelFunc, cuModule, fname));
	    name2CUfunc.emplace(fname, kernelFunc);
	}
    }
}

void parseJson()
{
    std::string jsonFile = "./Interposer/klist_caffe.json";
    std::string jsonStr = read_file(jsonFile);
    //std::cerr<<"File name: "<<jsonFile<<std::endl;
    //std::cerr<<"jsonStr: "<<jsonStr<<std::endl;
    rapidjson::Document document;
    document.Parse(jsonStr.c_str());
    if (document.IsObject()) {
	auto const& sm86 = document["86"];
        auto const& sm80 = document["80"];
	if (sm80.IsObject() || sm86.IsObject()) {
	    std::cerr<<"80!!"<<std::endl;
	    for (auto it = sm80.MemberBegin(); it != sm80.MemberEnd(); ++it) {
	        for (auto const& kernel : it->value.GetArray()) {
		    auto const& name = kernel["name"].GetString();
		    auto const& params = kernel["params"].GetArray();
		    const rapidjson::SizeType paramsSize = params.Size();
		    kernelMap.emplace(name, std::make_pair(it->name.GetString(), paramsSize));
		}
	    }
	    for (auto const& ptx : sm80.GetObject()) {
                auto const& ptx_filename = ptx.name.GetString();
                auto const& kernels = ptx.value.GetArray();

                for (auto const& kernel : kernels) {
                    auto const& name = kernel["name"].GetString();
                    PTXMap[ptx_filename].push_back(name);
                }
	    }
	    std::cerr<<"86!!"<<std::endl;

	    for (auto it = sm86.MemberBegin(); it != sm86.MemberEnd(); ++it) {
	        for (auto const& kernel : it->value.GetArray()) {
		    auto const& name = kernel["name"].GetString();
		    auto const& params = kernel["params"].GetArray();
		    const rapidjson::SizeType paramsSize = params.Size();
		    kernelMap86.emplace(name, std::make_pair(it->name.GetString(), paramsSize));
		}
	    }
	    for (auto const& ptx : sm86.GetObject()) {
                auto const& ptx_filename = ptx.name.GetString();
                auto const& kernels = ptx.value.GetArray();

                for (auto const& kernel : kernels) {
                    auto const& name = kernel["name"].GetString();
                    PTXMap86[ptx_filename].push_back(name);
                }
	    }
	}else{
		std::cerr<<"Not supported SM! "<<__LINE__<<std::endl;
	}
	if (sm80.IsObject() && sm86.IsObject()){
		std::cerr<<"Merge maps!! "<<std::endl;
		kernelMap.insert(kernelMap86.begin(), kernelMap86.end());
		PTXMap.insert(PTXMap86.begin(), PTXMap86.end());
	}

    }//isObject
}
void __cudaRegisterFunction(void **fatCubinHandle,
                const char *hostFun,
                char *deviceFun,
                const char *deviceName,
                int thread_limit,
                uint3 *tid,
                uint3 *bid,
                dim3 *bDim,
                dim3 *gDim,
                int *wSize){
    ptr2CUfunc_param.reserve(7000);
    //std::cerr<<"firstRegisterFunc: "<<firstRegisterFunc<<std::endl;
    if (firstRegisterFunc == 0){
        parseJson();
	std::cerr<<"Done with Parse JSON"<<std::endl;
	createModuleMap();
	std::cerr<<"Done with Module Map"<<std::endl;
	createFunctionMap();
	std::cerr<<"Done with Function Map"<<std::endl;
	firstRegisterFunc=1;
    }
    //fprintf(stderr,"Intercepted register function\n");
    void (*__cudaRegisterFunction_original) (void **, const char*, char*, const char*, int, uint3*, uint3*, dim3*, dim3*, int*) = (void (*)(void **, const char*, char*, const char*, int, uint3*, uint3*, dim3*, dim3*, int*)) dlsym(RTLD_NEXT, "__cudaRegisterFunction");
    __cudaRegisterFunction_original(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);

    //Store for every ptr the kernel name
    ptr2name[hostFun] = (char*)deviceName;
    for (const auto& entry : ptr2name) {
        const void* key = entry.first;
	const char* name = entry.second;
	auto it = kernelMap.find(name);
	if (it != kernelMap.end()) {
	    int param = it->second.second;
	    CUfunction cuFunc = name2CUfunc[name];
	    std::pair<CUfunction, int> pair(cuFunc, param);
	    ptr2CUfunc_param.emplace(key, pair);
	}
    }
//printf("hostfun:%p, deviceName: %s\n", hostFun, deviceName);
}
//static cudaError_t (*orig_cudaLaunchKernel)(const void*, dim3, dim3, void**, size_t, cudaStream_t) = NULL;
#ifdef DEBUG
// Function to look up a value in the map and return an error if the key does not exist
int lookup_value(const std::unordered_map<const void*, std::pair<CUfunction, int>>& map, const void* key, std::pair<CUfunction, int>& result) {
  auto iter = map.find(key);
  if (iter == map.end()) {
    return -1;  // Return an error code if the key is not found
  } else {
    result = iter->second;
    return 0;
  }
}
#endif
#ifdef TIMERS
unsigned long long results[40];
int count = 0;
#endif

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, 
		void **args, size_t sharedMem, cudaStream_t stream) {
#ifdef TIMERS 
    //s_map_chrono = std::chrono::high_resolution_clock::now();
    s_index_map1 = rdtsc();
#endif

    CUfunction kernelFunc;
    int argsNum = 0;
//    try {
	    // Get the pair associated with the key
	    const auto& pair = ptr2CUfunc_param.at(func);
	    // Access the kernel name in the tupple
	    kernelFunc = std::get<0>(pair);
	    // Access the number of args
	    argsNum = std::get<1>(pair);
 /*    } catch (std::out_of_range& e) {
        std::cout << "Error: " << e.what() << ". Key " << func<<" found in the map." << std::endl;
	abort();
    } 
*/
#ifdef TIMERS 
    e_index_map1 = rdtsc();
    //e_map_chrono = std::chrono::high_resolution_clock::now();
    s_args = rdtsc();
#endif    

printf("Intercepted launch of kernel %p with grid size %d x %d x %d and block size %d x %d x %d - NEW KRNL: %p\n", func, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, kernelFunc);

    //long int deviceExtraArgument = 0xfffffffffffffff;
    long int deviceExtraArgument = partitions[partition_index].mask;
#ifdef DEBUG
    std::cerr<<std::hex<<"Mask: "<<deviceExtraArgument<<std::dec<<std::endl;
#endif
    void **newArgs = (void **)malloc(sizeof(void *) * (argsNum + 1));
    memcpy(newArgs, args, sizeof(void *) * argsNum);
    newArgs[argsNum] = &deviceExtraArgument;

#ifdef TIMERS
    e_args = rdtsc();
    s_compute = rdtsc();
#endif

    CUresult err = cuLaunchKernel(kernelFunc, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, sharedMem, stream, newArgs, NULL);

#ifdef TIMERS
    CUDA_ERROR_FATAL(cuCtxSynchronize());
    e_compute = rdtsc();
    results[count] = e_compute - s_compute;
    count++;
#endif 
   if (err == CUDA_SUCCESS)
        return cudaSuccess;
    else
	return cudaErrorUnknown;

}

cudaError_t (*cudaFree_original) (void *);
cudaError_t cudaFree(void *devPtr) {
    if (!cudaFree_original) {
        cudaFree_original = (cudaError_t (*)(void *)) dlsym(RTLD_NEXT, "cudaFree");
    }
    // Call the original cudaFree function
    //return cudaFree_original(devPtr);
    return cudaSuccess;
}
#ifdef TIMERS
cudaError_t (*cudaDeviceReset_original) ();
cudaError_t cudaDeviceReset() {
    std::cerr << "Intercept cudaDeviceReset" << std::endl;
    if (!cudaDeviceReset_original) {
        cudaDeviceReset_original = (cudaError_t (*)()) dlsym(RTLD_NEXT, "cudaDeviceReset");
    }
    printf("Index cycles: %.3lu \n", (e_index_map1 - s_index_map1));
    printf("Args cycles: %.3lu \n", (e_args - s_args));

    unsigned long long sum = 0;
    std::cerr<<"Count: "<<count<<std::endl;
    for (int i=0; i<count; i++){
        std::cerr<<"count "<<i<<" cycles: "<<results[i]<<std::endl;
        sum += results[i];
    }

    printf("PTX cycles: %.3lu \n", (e_compute - s_compute));
    printf("AVG PTX cycles: %.3lu \n", sum/count);
    
    std::cerr<<"===================================="<<std::endl;                                                                                  
 
    // Call the original cudaFree function
    return cudaDeviceReset_original();
}
#endif
//cudaError_t (*cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_original) (int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags);

/*cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags( int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags) 
{
   printf("Intercept cudaOccupancyMaxActiveBlocks\n");
   cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_original = (cudaError_t (*)(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags)) dlsym(RTLD_NEXT, "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
   // Call the original cudaFree function
   return cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_original(numBlocks, func, blockSize, dynamicSMemSize, flags);
}*/
/*
CUresult (*cuDeviceGetAttribute_original) (int*, CUdevice_attribute, CUdevice);

CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev)
{
   printf("Intercept cuDeviceGetAttribute!!!!!!!!\n");
    if (!cuDeviceGetAttribute_original)
    {
        cuDeviceGetAttribute_original = (CUresult (*)(int*, CUdevice_attribute, CUdevice)) dlsym(RTLD_NEXT, "cuDeviceGetAttribute");
    }

    return cuDeviceGetAttribute_original(pi, attrib, dev);
}*/
/*
nvrtcResult nvrtcCreateProgram(const nvrtcProgram* prog, const char* src, const char* name, int numHeaders, const char** headers, const char** includeNames) {
    // Add your interception code here

    // Call the original function
    nvrtcResult res = nvrtcCreateProgram(prog, src, name, numHeaders, headers, includeNames);

    // Add any additional interception code here

    return res;
}*/
}
