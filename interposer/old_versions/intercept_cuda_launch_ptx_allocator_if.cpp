#include "rapidjson/document.h"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cublas_v2.h>
#include <cuda.h>         //Driver API
#include <cuda_runtime.h> //Runtime API
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdio.h>
#include <streambuf>
#include <string>
#include <time.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>
// #define DEBUG
#define NO_NEW_PTX
#define PARTITION_NUM 1
//#define PATH "../../../../../Interposer"
#define PATH "../Interposer"
// #define TIMERS
#define RED "\033[1;31m"
#define RESET "\033[0m"
int64_t s_compute;
uint64_t e_compute;

uint64_t s_index_map1;
uint64_t e_index_map1;

uint64_t s_args;
uint64_t e_args;

std::chrono::high_resolution_clock::time_point s_map_chrono;
std::chrono::high_resolution_clock::time_point e_map_chrono;

using namespace std;

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
  const char *err_str = nullptr;
  if (err != CUDA_SUCCESS) {
    cuGetErrorString(err, &err_str);
    std::cerr << RED << func << " error : " << RESET << err_str << " " << err
              << std::endl;
    std::cerr << "\t" << file << RED << " Failed at " << RESET << line
              << std::endl;
    exit(1);
  }
}

#define CUDA_ERROR_FATAL_RUNTIME(err)                                          \
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

typedef struct {
  size_t alloc_size;
  void *base_alloc_ptr;
  void *end_alloc_ptr;

} Allocation;

typedef struct {
  void *partition_ptr;
  size_t partition_size;
  size_t partition_free_space;
  pid_t pid;
  int bits2shift;
  long int mask;
  void* low_addr;
  void* high_addr;
  void* zero;
  // std::vector <Allocation> partition_allocations;
  int allocation_cnt = 0;
} Partition;

extern "C" {
// Map with key: krnl_ptr and values a pair of: CUfunction, #params
static std::unordered_map<const void *, std::pair<CUfunction, int>>
    *ptr2CUfunc_param;

int firstRegisterFunc = 0;
int firstcudaLaunch = 0;
std::string read_file(const std::string &filename) {
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
static Partition *partitions = nullptr;
static pid_t prev_pid = 0;
static int prev_partition_index = 0;
static void *prev_end_ptr = NULL;
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
  // std::cout << "Original address: 0x" << std::hex << addr << std::endl;
  uintptr_t aligned_addr = addr & ~(alignment - 1);
  // std::cout << "Aligned address-1: 0x" << std::hex << aligned_addr-1 <<
  // std::endl;
  return aligned_addr - alignment;
}

uint64_t get_partition_size(uint32_t n, uint64_t gpu_memory_size) {
  // Compute the size of each partition
  const uint64_t partition_size = gpu_memory_size / n;

  // Determine the power of two that is greater or equal to the partition size
  const uint64_t pow2_partition_size = pow(2, floor(log2(partition_size)));
  // const uint64_t pow2_partition_size = 1ull << (64 -
  // __builtin_clzll(partition_size));

  // Create the partitions by setting the appropriate bits
  uint64_t remaining_memory = gpu_memory_size;
  for (uint32_t i = 0; i < n; ++i) {
    const uint64_t partition_size =
        std::min(pow2_partition_size, remaining_memory);
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

  return count - 1;
}
unsigned long long int extractPartition(uintptr_t address, size_t partitionSize)
{
   // Create bitmask based on partition size
   unsigned long long bitmask = (1ULL << (partitionSize * 4)) - 1;
   unsigned long long result = address & bitmask;
   return result;
}

size_t calculateDigitsForPartitionSize(unsigned long long addressSpaceSize)
{
    unsigned int numberOfDigits =
            static_cast<unsigned int>(std::ceil(std::log2(addressSpaceSize)));
    std::cout << "Number of digits required: " << numberOfDigits -1  <<std::endl;
    unsigned int numberOfDigits48 = numberOfDigits - 1;
    unsigned int numberOfHexDigits =
            static_cast<unsigned int>((static_cast<double>(numberOfDigits48) / 4));
    std::cout <<" Hex: "<<numberOfHexDigits<< std::endl;
    return numberOfHexDigits;
}
cudaError_t (*cudaMalloc_original)(void **, size_t);
cudaError_t cudaMalloc(void **devPtr, size_t size) {
  size_t partition_size;
#ifdef DEBUG
  std::cerr << "Intercept Malloc!!" << std::endl;
#endif
  pid_t pid = getpid();
  // std::cerr<<"PID: "<<pid<<" PREV: "<<prev_pid<<std::endl;
  if (!cudaMalloc_original) {
    cudaMalloc_original =
        (cudaError_t(*)(void **, size_t))dlsym(RTLD_NEXT, "cudaMalloc");
  }
  if (partitions == nullptr) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    size_t total_memory = deviceProp.totalGlobalMem;
    partition_size = get_partition_size(PARTITION_NUM, total_memory);
//#ifdef DEBUG
    std::cerr << "Total memory: " << total_memory
              << " ,Partition size: " << partition_size << std::endl;
//#endif
    partitions = new Partition[PARTITION_NUM];
    for (int i = 0; i < PARTITION_NUM; i++) {
      partitions[i].partition_size = partition_size;
      partitions[i].pid = -1;
      void *tmp_ptr;
      CUDA_ERROR_FATAL_RUNTIME(cudaMalloc_original(&tmp_ptr, partition_size));
      //	    std::cerr<<"i: "<<i<<" allocation size: "<<partition_size
      //		    <<" returned ptr: "<<tmp_ptr<<std::endl;
      partitions[i].partition_ptr = align(partition_size, tmp_ptr);
      uintptr_t highest_ptr =
          (uintptr_t)partitions[i].partition_ptr + partitions[i].partition_size;

      // uintptr_t mask = get_mask((uintptr_t)partitions[i].partition_ptr, 1);
      uintptr_t mask = get_mask(highest_ptr, 1);
      std::cerr << "Highest: " << std::hex << highest_ptr << " ,Mask: " << mask
                << std::dec << " ,Diff: " << highest_ptr - mask << std::endl;

#ifdef DEBUG
      if (highest_ptr - mask != 1) {
        std::cerr << "mask and partition has diff greater than 1!" << std::endl;
        abort();
      }
#endif
      partitions[i].low_addr = partitions[i].partition_ptr;
      partitions[i].high_addr = (void*)highest_ptr;
      partitions[i].partition_free_space = partition_size;
      int digits = calculateDigitsForPartitionSize(partition_size);
      partitions[i].zero = (void*)extractPartition(mask, digits);

      partitions[i].mask = mask;
      partitions[i].partition_free_space = partition_size;
    }
  }
  // Find the next free partition to allocate based on pid
  // partitions[0].pid = 3254;
  for (int i = 0; i < PARTITION_NUM; i++) {
    if (pid != prev_pid) {
      // check if partition is not assigned to another process
      if (partitions[i].pid == -1) {
        partition_index = i;
        prev_partition_index = partition_index;
        break;
      }
    } else {
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
  Partition *partition = &partitions[partition_index];
  partition->pid = pid;
#ifdef DEBUG
  std::cerr << "Selected partition: " << partition_index
            << " ,pid: " << partitions[partition_index].pid << " ,free memory: "
            << partitions[partition_index].partition_free_space << std::endl;
#endif
  if (size < partitions[partition_index].partition_free_space) {
    partitions[partition_index].partition_free_space =
        partitions[partition_index].partition_free_space - size;
#ifdef DEBUG
    std::cerr << "Remaining space left: "
              << partitions[partition_index].partition_free_space << std::endl;
#endif
    int current_partition_allocations =
        partitions[partition_index].allocation_cnt;
    /*	std::cerr << "Current partition allocations: "
                    << current_partition_allocations <<" partition index: "
                    << partition_index << std::endl;*/
    Allocation allocation;
    // partition_allocations vector is empty!!!!
    // allocation =
    // partition->partition_allocations[current_partition_allocations];
    allocation.alloc_size = size;
    // std::cerr<<"allocation size: "<<allocation.alloc_size<<std::endl;
    // std::cerr<<"previous end ptr: "<<prev_end_ptr<<std::endl;
    //  If it is the first allocation
    if (prev_end_ptr == NULL)
      // Assign the partition ptr
      allocation.base_alloc_ptr = partition->partition_ptr;
    else
      // Assign the end ptr of the (power of two) previous segment
      allocation.base_alloc_ptr = align(256, prev_end_ptr);
    // allocation.base_alloc_ptr = align(256, prev_end_ptr);
    // Now the end ptr is the base + size of allocation
    allocation.end_alloc_ptr =
        (void *)(((unsigned long long)allocation.base_alloc_ptr) + size);
    // store the allocation end ptr to the previous ptr (to be used in next
    // allocations)
    prev_end_ptr = allocation.end_alloc_ptr;
    // increase the segment num of the current parition
    partitions[partition_index].allocation_cnt++;
    // return the device pointer
    *devPtr = allocation.base_alloc_ptr;

    void *highest_addr_partition =
        (void *)(((unsigned long long)partitions[partition_index]
                      .partition_ptr) +
                 partitions[partition_index].partition_size);
#ifdef DEBUG
    std::cerr << "Partition-> Lowest addr: "
              << partitions[partition_index].partition_ptr
              << " , Highest addr: " << std::hex << highest_addr_partition
              <<" low_addr: "<< partitions[partition_index].low_addr
              <<" hig_addr: "<< partitions[partition_index].high_addr
              <<" zero: "<< partitions[partition_index].zero
              << " , Size: " << partitions[partition_index].partition_size
              <<std::dec<< std::endl;

    if (*devPtr > highest_addr_partition) {
      std::cerr << "Allocation is outside the partition!" << std::endl;
      abort();
    }

    std::cerr << "---------------------------------------" << std::endl;
#endif
    // Store the allocation
    // partition->partition_allocations[current_partition_allocations] =
    // allocation;
    prev_pid = pid;
    return cudaSuccess;

  } else {
    std::cerr << "There is no space left in the " << partition_index
              << " partition"
              << " for the requested allocation " << size << " remaining space "
              << partitions[partition_index].partition_free_space << std::endl;
    return cudaErrorMemoryAllocation;
  }
}

std::unordered_map<std::string, CUmodule> createModuleMap(
    std::unordered_map<std::string, std::vector<std::string>> PTXMap) {
  //  key: ptx, value cumodule
  std::unordered_map<std::string, CUmodule> ptx2module;
  CUdevice cuDevice;
  CUcontext cuContext;
  int devID = 0;
  CUDA_ERROR_FATAL(cuInit(0));
  CUDA_ERROR_FATAL(cuDeviceGet(&cuDevice, devID));
  CUDA_ERROR_FATAL(cuCtxCreate(&cuContext, 0, cuDevice));

  CUmodule cuModule;

  for (const auto &entry : PTXMap) {
    const std::string &ptx = entry.first;

    std::string path = PATH"/mod_if_caffe_libs/" + ptx;
    std::ifstream my_file(path);
    //std::cerr<<"Path: "<<path<<std::endl;
    std::string my_ptx((std::istreambuf_iterator<char>(my_file)),
                       std::istreambuf_iterator<char>());

    //std::cerr<<"my_ptx.c_str(): "<<my_ptx.c_str()<<path<<std::endl;
    CUresult err = cuModuleLoadData(&cuModule, my_ptx.c_str());
    if (err != CUDA_SUCCESS) {
      const char *err_str = nullptr;
      cuGetErrorString(err, &err_str);
      std::cerr << RED << " cuModuleLoadData error : " << RESET << err_str
                << " " << err << " for PTX: " << path << std::endl;
      abort();
    }
    ptx2module.emplace(ptx, cuModule);
  }
  // std::cerr<<"End: "<<__func__<<std::endl;
  return ptx2module;
}

std::unordered_map<std::string, CUfunction> createFunctionMap(
    std::unordered_map<std::string, std::vector<std::string>> PTXMap,
    std::unordered_map<std::string, CUmodule> ptx2module) {
  std::unordered_map<std::string, CUfunction> name2CUfunc;
  // std::cerr<<"Start: "<<__func__<<std::endl;
  CUfunction kernelFunc;
  for (auto &entry : PTXMap) {
    const std::string &ptxName = entry.first;
    // std::cerr<<"PTX: "<<ptxName<<std::endl;
    CUmodule cuModule = ptx2module.at(ptxName);
    std::vector<std::string> &values = entry.second; // get the vector of values
    for (std::string &krnl : values) {
      const char *fname = krnl.c_str();
      // std::cerr<<"Function name: "<<fname<<std::endl;
      CUDA_ERROR_FATAL(cuModuleGetFunction(&kernelFunc, cuModule, fname));
      name2CUfunc.emplace(fname, kernelFunc);
    }
  }
#ifdef DEBUG 
  std::cerr << "Open file nameFunc!! " << std::endl;
  std::ofstream outFile("nameFunc.txt");
  if (!outFile.is_open()) {
    std::cerr << "Failed to open the file for writing." << std::endl;
    abort();
  }
  // Write the contents of the map to the file
  for (const auto &kv : name2CUfunc) {
    outFile << kv.first << " , " << kv.second << std::endl;
  }
  outFile.close();
#endif
  // std::cerr<<"Done: "<<__func__<<std::endl;
  return name2CUfunc;
}

// std::unordered_map<std::string, std::pair<std::string, int>> parseJson()
// void parseJson()
void parseJson(
    std::unordered_map<std::string, std::pair<std::string, int>> &kernelMap,
    std::unordered_map<std::string, std::vector<std::string>> &PTXMap) {
  // std::cerr<<__func__<<std::endl;
  std::string jsonFile = PATH"/klist_caffe.json";
  //std::string jsonFile = PATH "/klist_caffe_short.json";
  std::string jsonStr = read_file(jsonFile);
  std::cerr << "File name: " << jsonFile << std::endl;
  // std::cerr<<"jsonStr: "<<jsonStr<<std::endl;
  rapidjson::Document document;
  document.Parse(jsonStr.c_str());
  if (document.IsObject()) {
    // std::cerr<<"Doc is obj!!!"<<std::endl;
    if (document.HasMember("80")) {
      auto const &sm80 = document["80"];
      if (sm80.IsObject()) {
        std::cerr << "SM 80" << std::endl;
        for (auto it = sm80.MemberBegin(); it != sm80.MemberEnd(); ++it) {
          for (auto const &kernel : it->value.GetArray()) {
            auto const &name = kernel["name"].GetString();
            auto const &params = kernel["params"].GetArray();
            const rapidjson::SizeType paramsSize = params.Size();
            kernelMap.emplace(name,
                              std::make_pair(it->name.GetString(), paramsSize));
          }
        }
        for (auto const &ptx : sm80.GetObject()) {
          auto const &ptx_filename = ptx.name.GetString();
          auto const &kernels = ptx.value.GetArray();

          for (auto const &kernel : kernels) {
            auto const &name = kernel["name"].GetString();
            PTXMap[ptx_filename].push_back(name);
          }
        }
      }
    }
    if (document.HasMember("86")) {
      std::unordered_map<std::string, std::pair<std::string, int>> kernelMap86;
      std::unordered_map<std::string, std::vector<std::string>> PTXMap86;
      std::cerr << "SM 86" << std::endl;
      auto const &sm86 = document["86"];
      if (sm86.IsObject()) {
        for (auto it = sm86.MemberBegin(); it != sm86.MemberEnd(); ++it) {
          for (auto const &kernel : it->value.GetArray()) {
            auto const &name = kernel["name"].GetString();
            auto const &params = kernel["params"].GetArray();
            const rapidjson::SizeType paramsSize = params.Size();
            kernelMap86.emplace(
                name, std::make_pair(it->name.GetString(), paramsSize));
          }
        }
        for (auto const &ptx : sm86.GetObject()) {
          auto const &ptx_filename = ptx.name.GetString();
          auto const &kernels = ptx.value.GetArray();

          for (auto const &kernel : kernels) {
            auto const &name = kernel["name"].GetString();
            PTXMap86[ptx_filename].push_back(name);
          }
        }
      }
      std::cerr << "Merge maps!! " << std::endl;
      kernelMap.insert(kernelMap86.begin(), kernelMap86.end());
      PTXMap.insert(PTXMap86.begin(), PTXMap86.end());
    }
    if (document.HasMember("75")) {
      std::unordered_map<std::string, std::pair<std::string, int>> kernelMap75;
      std::unordered_map<std::string, std::vector<std::string>> PTXMap75;

      auto const &sm75 = document["75"];
      if (sm75.IsObject()) {
        std::cerr << "SM 75" << std::endl;
        for (auto it = sm75.MemberBegin(); it != sm75.MemberEnd(); ++it) {
          for (auto const &kernel : it->value.GetArray()) {
            auto const &name = kernel["name"].GetString();
            auto const &params = kernel["params"].GetArray();
            const rapidjson::SizeType paramsSize = params.Size();
            // std::cerr<<name<<std::endl;
            kernelMap75.emplace(
                name, std::make_pair(it->name.GetString(), paramsSize));
          }
        }
        for (auto const &ptx : sm75.GetObject()) {
          auto const &ptx_filename = ptx.name.GetString();
          auto const &kernels = ptx.value.GetArray();

          for (auto const &kernel : kernels) {
            auto const &name = kernel["name"].GetString();
            PTXMap75[ptx_filename].push_back(name);
          }
        }
      }
      std::cerr << "Merge maps 75!! " << std::endl;
      kernelMap.insert(kernelMap75.begin(), kernelMap75.end());
      PTXMap.insert(PTXMap75.begin(), PTXMap75.end());

    } else {
      std::cerr << "Not supported architecture! Abort! " << __LINE__
                << std::endl;
      abort();
    }

  } // isObject
  else {
    std::cerr << "File not an object. Abort! " << __LINE__ << std::endl;
    abort();
  }
}
void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                            char *deviceFun, const char *deviceName,
                            int thread_limit, uint3 *tid, uint3 *bid,
                            dim3 *bDim, dim3 *gDim, int *wSize) {
  // key: krnl name, values: ptx, params
  static std::unordered_map<std::string, std::pair<std::string, int>> kernelMap;

  // key: ptx, values: krnl name
  std::unordered_map<std::string, std::vector<std::string>> PTXMap;
  static std::unordered_map<std::string, CUfunction> CUfuncMap;

  //std::cerr<<"firstRegisterFunc: "<<firstRegisterFunc<<std::endl;
  if (firstRegisterFunc == 0) {
    ptr2CUfunc_param =
        new std::unordered_map<const void *, std::pair<CUfunction, int>>();
    ptr2CUfunc_param->reserve(10);
    // func2Cubin.reserve(200);

    parseJson(kernelMap, PTXMap);
    std::cerr << "PTX Map size: " << PTXMap.size() << std::endl;
    std::cerr << "Kernel Map size: " << kernelMap.size() << std::endl;
    std::cerr<<" ===== Done parse json ====="<<std::endl;

    static auto moduleMap = createModuleMap(PTXMap);
    std::cerr << "Module Map size: " << moduleMap.size() << std::endl;
    std::cerr<<" ===== Done module map ====="<<std::endl;

    CUfuncMap = createFunctionMap(PTXMap, moduleMap);
    std::cerr << "CUFUNC Map size: " << CUfuncMap.size() << std::endl;
    std::cerr<<" ===== Done cufunction ====="<<std::endl;

    firstRegisterFunc = 1;
  }
  // fprintf(stderr,"Intercepted register function\n");
  void (*__cudaRegisterFunction_original)(void **, const char *, char *,
                                          const char *, int, uint3 *, uint3 *,
                                          dim3 *, dim3 *, int *) =
      (void (*)(void **, const char *, char *, const char *, int, uint3 *,
                uint3 *, dim3 *, dim3 *, int *))dlsym(RTLD_NEXT,
                                                      "__cudaRegisterFunction");
  __cudaRegisterFunction_original(fatCubinHandle, hostFun, deviceFun,
                                  deviceName, thread_limit, tid, bid, bDim,
                                  gDim, wSize);

  const void *key = hostFun;
  const char *name = (char *)deviceName;
  // std::cerr << "Key: " << key << " name " << name << std::endl;
  //  std::cerr<<"-> kernel map size: "<<kernelMap.size()<<std::endl;
  printf("hostfun:%p, deviceName: %s\n", hostFun, deviceName);
  auto it = kernelMap.find(name);

  if (it != kernelMap.end()) {
    // std::cerr<<"Name is found!!"<<std::endl;
    int param = it->second.second;
    CUfunction cuFunc = CUfuncMap.at(name);
    // std::cerr<<"CuFunc: "<<cuFunc<<std::endl;
    std::pair<CUfunction, int> pair(cuFunc, param);
    ptr2CUfunc_param->emplace(key, pair);
  }
  /*else {
    std::cout << "Ptr: " << key << " Name: " << name
              << " not found in kernelMap. Stored in map!" << std::endl;
    // func2Cubin.emplace(key,name);
    // abort();
  }*/

  // std::cerr<<"ptr2name size: "<<ptr2name.size()<<std::endl;
  // std::cerr << "ptr2CUfunc_param size: " << ptr2CUfunc_param->size()
  //          << " ,ptr: " << &ptr2CUfunc_param << std::endl;
  // printf("hostfun:%p, deviceName: %s\n", hostFun, deviceName);
}

#ifdef TIMERS
unsigned long long results[40];
int count = 0;
#endif
//#define KRNLCNT
static unsigned long long kernelCount = 0;
cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem,
                             cudaStream_t stream) {
#ifdef TIMERS
  // s_map_chrono = std::chrono::high_resolution_clock::now();
  s_index_map1 = rdtsc();
#endif

  CUfunction kernelFunc;
  int argsNum;
#ifdef DEBUG
  try {
#endif
    // Get the pair associated with the key
    const auto &pair = ptr2CUfunc_param->at(func);
    // Access the kernel name in the tupple
    kernelFunc = std::get<0>(pair);
    // Access the number of args
    argsNum = std::get<1>(pair);
#ifdef DEBUG
    std::cerr<<"----- Args num: "<<argsNum<<std::endl;
  } catch (std::out_of_range &e) {
    std::cout << "Error: " << e.what() << ". Key " << func
              << " found in the map." << std::endl;
    abort();
  }
  printf("Intercepted launch of kernel %p with grid size %d x %d x %d and block size %d x %d x %d - NEW KRNL: %p\n", func, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, kernelFunc);
#endif

//#endif
#ifdef KRNLCNT
  kernelCount++;
#endif
#ifdef DEBUG
  if (firstcudaLaunch == 0) {
    std::cerr << "Open file !! " << std::endl;
    std::ofstream outFile("registeredKernels.txt");
    if (!outFile.is_open()) {
      std::cerr << "Failed to open the file for writing." << std::endl;
      abort();
    }
    // Write the map elements to the file
    for (const auto &entry : *ptr2CUfunc_param) {
      const void *key = entry.first;
      CUfunction func = entry.second.first;
      int param = entry.second.second;
      outFile << key << " " << func << " " << param << std::endl;
    }
    outFile.close();
    firstcudaLaunch = 1;
  }
#endif 
#ifdef TIMERS
  e_index_map1 = rdtsc();
  // e_map_chrono = std::chrono::high_resolution_clock::now();
  s_args = rdtsc();
#endif
  //std::cerr << "Intercepted: " << func <<" args: "<<argsNum<< std::endl;
  //unsigned long int  deviceExtraArgument = 0xFFFFFFFFFFFFFFFFLL;
  void* deviceExtraArgument1 = partitions[partition_index].low_addr;
  void* deviceExtraArgument2 = partitions[partition_index].high_addr;
  void **newArgs = (void **)malloc(sizeof(void *) * (argsNum + 2));
  memcpy(newArgs, args, sizeof(void *) * argsNum);
  newArgs[argsNum] = (void *) &deviceExtraArgument1;
  newArgs[argsNum+1] = (void *) &deviceExtraArgument2;
#ifdef DEBUG 
  std::cerr << std::hex << "Zero: " << deviceExtraArgument1 << std::dec
            <<" Low: "<<deviceExtraArgument2<<std::endl; 
#endif
 
#ifdef TIMERS
  e_args = rdtsc();
  s_compute = rdtsc();
#endif

  CUresult err =
      cuLaunchKernel(kernelFunc, gridDim.x, gridDim.y, gridDim.z, blockDim.x,
                     blockDim.y, blockDim.z, sharedMem, stream, newArgs, NULL);

  CUDA_ERROR_FATAL(cuCtxSynchronize());
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

cudaError_t (*cudaFree_original)(void *);
cudaError_t cudaFree(void *devPtr) {
  if (!cudaFree_original) {
    cudaFree_original = (cudaError_t(*)(void *))dlsym(RTLD_NEXT, "cudaFree");
  }
  // Call the original cudaFree function
  // return cudaFree_original(devPtr);
  return cudaSuccess;
}
#ifdef KRNLCNT
cudaError_t (*cudaDeviceReset_original)();
cudaError_t cudaDeviceReset() {
  std::cerr << "Intercept cudaDeviceReset" << std::endl;
  std::cerr<<" Kernel: "<<kernelCount<<std::endl;
/*  if (!cudaDeviceReset_original) {
    cudaDeviceReset_original =
        (cudaError_t(*)())dlsym(RTLD_NEXT, "cudaDeviceReset");
  }
*/
#if 0
  printf("Index cycles: %.3lu \n", (e_index_map1 - s_index_map1));
  printf("Args cycles: %.3lu \n", (e_args - s_args));

  unsigned long long sum = 0;
  std::cerr << "Count: " << count << std::endl;
  for (int i = 0; i < count; i++) {
    std::cerr << "count " << i << " cycles: " << results[i] << std::endl;
    sum += results[i];
  }

  printf("PTX cycles: %.3lu \n", (e_compute - s_compute));
  printf("AVG PTX cycles: %.3lu \n", sum / count);

  std::cerr << "====================================" << std::endl;
#endif
  // Call the original cudaFree function
  return cudaSuccess;
}
#endif
// cudaError_t
// (*cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_original) (int*
// numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned
// int flags);

/*cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags( int*
numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int
flags)
{
   printf("Intercept cudaOccupancyMaxActiveBlocks\n");
   cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_original =
(cudaError_t (*)(int* numBlocks, const void* func, int blockSize, size_t
dynamicSMemSize, unsigned int flags)) dlsym(RTLD_NEXT,
"cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
   // Call the original cudaFree function
   return
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_original(numBlocks, func,
blockSize, dynamicSMemSize, flags);
}*/
/*
CUresult (*cuDeviceGetAttribute_original) (int*, CUdevice_attribute, CUdevice);

CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev)
{
   printf("Intercept cuDeviceGetAttribute!!!!!!!!\n");
    if (!cuDeviceGetAttribute_original)
    {
        cuDeviceGetAttribute_original = (CUresult (*)(int*, CUdevice_attribute,
CUdevice)) dlsym(RTLD_NEXT, "cuDeviceGetAttribute");
    }

    return cuDeviceGetAttribute_original(pi, attrib, dev);
}*/
/*
nvrtcResult nvrtcCreateProgram(const nvrtcProgram* prog, const char* src, const
char* name, int numHeaders, const char** headers, const char** includeNames) {
    // Add your interception code here

    // Call the original function
    nvrtcResult res = nvrtcCreateProgram(prog, src, name, numHeaders, headers,
includeNames);

    // Add any additional interception code here

    return res;
}*/
}
