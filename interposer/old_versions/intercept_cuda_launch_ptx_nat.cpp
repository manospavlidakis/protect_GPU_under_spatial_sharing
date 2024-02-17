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
//#define PATH "../../../../../Interposer"
#define PATH "../Interposer"
//#define DEBUG
#define NO_NEW_PTX
//#define TIMERS
#define RED "\033[1;31m"
#define RESET "\033[0m"
uint64_t s_compute;
uint64_t e_compute;

uint64_t s_index_map1;
uint64_t e_index_map1;


uint64_t s_args;
uint64_t e_args;

std::chrono::high_resolution_clock::time_point s_map_chrono;
std::chrono::high_resolution_clock::time_point e_map_chrono;
/*
std::chrono::high_resolution_clock::time_point s_1;
std::chrono::high_resolution_clock::time_point e_1;

std::chrono::high_resolution_clock::time_point s_2;
std::chrono::high_resolution_clock::time_point e_2;
*/
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
extern "C"{

//Map with key: krnl_ptr and values a pair of: CUfunction, #params
static std::unordered_map<const void*, std::pair<CUfunction, int>> *ptr2CUfunc_param;
//static std::unordered_map<const void*, std::string> func2Cubin;

int firstRegisterFunc = 0;
int firstLaunch = 0;
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

std::unordered_map<std::string, CUmodule> createModuleMap(
		std::unordered_map<std::string, std::vector<std::string>> PTXMap) {
    std::cerr<<"--> Umodified PTX "<<std::endl;
    // key: ptx, value cumodule
    std::unordered_map<std::string, CUmodule> ptx2module;
    CUdevice cuDevice; 
    CUcontext cuContext;
    int devID = 0;
    CUDA_ERROR_FATAL(cuInit(0));
    CUDA_ERROR_FATAL(cuDeviceGet(&cuDevice, devID));
    CUDA_ERROR_FATAL(cuCtxCreate(&cuContext, 0, cuDevice));

    CUmodule cuModule;

    for (const auto& entry : PTXMap) {
        const std::string& ptx = entry.first;

	//std::string path = PATH"/tmp_caffe/" + ptx;  
	std::string path = PATH"/caffe_libs/" + ptx;  
	//std::string path = "../Interposer/caffe_libs/" + ptx;  
	//std::cerr<<"Path: "<<path<<std::endl;
	//std::string path = "../Interposer/native_cublas/" + ptx;  
	//std::string path = "../../../../../Interposer/caffe_libs/" + ptx;  
	std::ifstream my_file(path);
	std::string my_ptx((std::istreambuf_iterator<char>(my_file)), 
			std::istreambuf_iterator<char>());

	CUresult err = cuModuleLoadData(&cuModule, my_ptx.c_str());
	if (err != CUDA_SUCCESS) {
            const char* err_str = nullptr;
	    cuGetErrorString(err, &err_str);
	    std::cerr << RED << " cuModuleLoadData error : " 
		<< RESET << err_str << " " << err << " for PTX: "
                << my_ptx.c_str() << std::endl;
	    abort();
	}
	ptx2module.emplace(ptx, cuModule);
    }
    //std::cerr<<"End: "<<__func__<<std::endl;
    return ptx2module;
}

std::unordered_map<std::string, CUfunction> createFunctionMap(
		std::unordered_map<std::string, std::vector<std::string>> PTXMap, 
		std::unordered_map<std::string, CUmodule> ptx2module) 
{
    std::unordered_map<std::string, CUfunction> name2CUfunc;
    //std::cerr<<"Start: "<<__func__<<std::endl;
    CUfunction kernelFunc;
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
    //std::cerr<<"Done: "<<__func__<<std::endl;
    return name2CUfunc;
}

//std::unordered_map<std::string, std::pair<std::string, int>> parseJson()
//void parseJson()
void parseJson(std::unordered_map<std::string, std::pair<std::string, int>>& kernelMap, 
		std::unordered_map<std::string, std::vector<std::string>>& PTXMap) {
    //std::cerr<<__func__<<std::endl;
    //std::string jsonFile = PATH"/tmp_klist_caffe.json";
    std::string jsonFile = PATH"/klist_caffe.json";
    //std::string jsonFile = "../Interposer/klist_caffe.json";
    //std::string jsonFile = "../../../../../Interposer/klist_caffe.json";
    std::string jsonStr = read_file(jsonFile);
    //std::cerr<<"File name: "<<jsonFile<<std::endl;
    //std::cerr<<"jsonStr: "<<jsonStr<<std::endl;
    rapidjson::Document document;
    document.Parse(jsonStr.c_str());
    if (document.IsObject()) {
	//std::cerr<<"Doc is obj!!!"<<std::endl;
	if (document.HasMember("80")) {
            auto const& sm80 = document["80"];
	    if (sm80.IsObject()) {
	        std::cerr<<"SM 80"<<std::endl;
		for (auto it = sm80.MemberBegin(); it != sm80.MemberEnd(); ++it) {
	             for (auto const& kernel : it->value.GetArray()) {
			  auto const& name = kernel["name"].GetString();
			  auto const& params = kernel["params"].GetArray();
			  const rapidjson::SizeType paramsSize = params.Size();
			  //std::cerr<<name<<std::endl;
				  //<<" it->name.GetString(): "<<it->name.GetString()
				  //<<" param size: "<<paramsSize<<std::endl;
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
	    }
	}
	if (document.HasMember("86")) {
	    std::unordered_map<std::string, std::pair<std::string, int>> kernelMap86;
	    std::unordered_map<std::string, std::vector<std::string>> PTXMap86;
            std::cerr<<"SM 86"<<std::endl;
	    auto const& sm86 = document["86"];
            if (sm86.IsObject()) {
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
	    }
	    std::cerr<<"Merge maps!! "<<std::endl;
	    kernelMap.insert(kernelMap86.begin(), kernelMap86.end());
	    PTXMap.insert(PTXMap86.begin(), PTXMap86.end());
 
	}
	if (document.HasMember("75")) {
	    std::unordered_map<std::string, std::pair<std::string, int>> kernelMap75;
	    std::unordered_map<std::string, std::vector<std::string>> PTXMap75;

            auto const& sm75 = document["75"];
	    if (sm75.IsObject()) {
	        std::cerr<<"SM 75"<<std::endl;
		for (auto it = sm75.MemberBegin(); it != sm75.MemberEnd(); ++it) {
	             for (auto const& kernel : it->value.GetArray()) {
			  auto const& name = kernel["name"].GetString();
			  auto const& params = kernel["params"].GetArray();
			  const rapidjson::SizeType paramsSize = params.Size();
			  //std::cerr<<name<<std::endl;
				  //<<" it->name.GetString(): "<<it->name.GetString()
				  //<<" param size: "<<paramsSize<<std::endl;
			  kernelMap75.emplace(name, std::make_pair(it->name.GetString(), paramsSize));
		     }
		}
		for (auto const& ptx : sm75.GetObject()) {
		     auto const& ptx_filename = ptx.name.GetString();
		     auto const& kernels = ptx.value.GetArray();

		     for (auto const& kernel : kernels) {
			  auto const& name = kernel["name"].GetString();
			  PTXMap75[ptx_filename].push_back(name);
		     }
		}
	    }
	    std::cerr<<"Merge maps 75!! "<<std::endl;
	    kernelMap.insert(kernelMap75.begin(), kernelMap75.end());
	    PTXMap.insert(PTXMap75.begin(), PTXMap75.end());

	}
	else{
	    std::cerr<<"Not supported architecture! Abort! "<<__LINE__<<std::endl;
	    abort();
	}
	
    }//isObject
    else{
        std::cerr<<"File not an object. Abort! "<<__LINE__<<std::endl;
	abort();
    }
}
	

void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                char *deviceFun, const char *deviceName, int thread_limit,
                uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize)
{
    // key: krnl name, values: ptx, params
    static std::unordered_map<std::string, std::pair<std::string, int>> kernelMap;

    // key: ptx, values: krnl name
    std::unordered_map<std::string, std::vector<std::string>> PTXMap;
    static std::unordered_map<std::string, CUfunction> CUfuncMap;
    
    //std::cerr<<"firstRegisterFunc: "<<firstRegisterFunc<<std::endl;
    if (firstRegisterFunc == 0){
	//std::cerr<<"INSIDEEE first!!!"<<std::endl;
        ptr2CUfunc_param = new std::unordered_map<const void*, std::pair<CUfunction, int>>();
	ptr2CUfunc_param->reserve(10);
	//func2Cubin.reserve(200);


        parseJson(kernelMap,PTXMap);
	std::cerr<<"PTX Map size: "<<PTXMap.size()<<std::endl;
	std::cerr<<"Kernel Map size: "<<kernelMap.size()<<std::endl;
	//std::cerr<<"Done with Parse JSON"<<std::endl;

	static auto moduleMap = createModuleMap(PTXMap);
	std::cerr<<"Module Map size: "<<moduleMap.size()<<std::endl;
	//std::cerr<<"Done with Module Map"<<std::endl;

	CUfuncMap = createFunctionMap(PTXMap, moduleMap);
	std::cerr<<"CUFUNC Map size: "<<CUfuncMap.size()<<std::endl;
	//std::cerr<<"Done with Function Map"<<std::endl;

    	firstRegisterFunc=1;
	/*for (const auto& entry : kernelMap) {
		std::string key = entry.first;
		std::string value1 = entry.second.first;
		int value2 = entry.second.second;
		std::cout << "Key: " << key << ", Value 1: " << value1 << ", Value 2: " << value2 << std::endl;
	}*/
    }
    //fprintf(stderr,"Intercepted register function\n");
    void (*__cudaRegisterFunction_original) (void **, const char*, char*, const char*, int, uint3*, uint3*, dim3*, dim3*, int*) = (void (*)(void **, const char*, char*, const char*, int, uint3*, uint3*, dim3*, dim3*, int*)) dlsym(RTLD_NEXT, "__cudaRegisterFunction");
    __cudaRegisterFunction_original(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);

    const void* key = hostFun;
    const char* name = (char*)deviceName;
    std::cerr<<"Key: "<<key<<" name "<<name<<std::endl;
    //std::cerr<<"-> kernel map size: "<<kernelMap.size()<<std::endl;
    //printf("hostfun:%p, deviceName: %s\n", hostFun, deviceName);
    auto it = kernelMap.find(name);

    if (it != kernelMap.end()) {
	//std::cerr<<"Name is found!!"<<std::endl;
	int param = it->second.second;
	CUfunction cuFunc = CUfuncMap.at(name);
	//std::cerr<<"CuFunc: "<<cuFunc<<std::endl;
	std::pair<CUfunction, int> pair(cuFunc, param);
        ptr2CUfunc_param->emplace(key, pair);
    } else {
        std::cout <<"Ptr: " <<key<< " Name: "<< name <<" not found in kernelMap. Stored in map!" << std::endl;	
	//func2Cubin.emplace(key,name);
	//abort();
    }

    //std::cerr<<"ptr2name size: "<<ptr2name.size()<<std::endl;
    std::cerr<<"ptr2CUfunc_param size: "<<ptr2CUfunc_param->size()
	    <<" ,ptr: "<<&ptr2CUfunc_param<<std::endl;
    //printf("hostfun:%p, deviceName: %s\n", hostFun, deviceName);
}
static cudaError_t (*orig_cudaLaunchKernel)(const void*, dim3, dim3, void**, size_t, cudaStream_t) = NULL;

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, 
		void **args, size_t sharedMem, cudaStream_t stream) {
/*   if (firstLaunch == 0){
       std::cerr<<"HERRRRRRRRRRRRREEEEEEEEEEEE. func2Cubin size: "<<func2Cubin.size()<<std::endl;
       for (const auto& entry : func2Cubin) {
	   std::cout << "Ptr: " << entry.first << ", Name: " << entry.second << std::endl;
       }  
       firstLaunch = 1;
   }*/
#ifdef NO_NEW_PTX

#ifdef TIMERS 
    //s_map_chrono = std::chrono::high_resolution_clock::now();
    s_index_map1 = rdtsc();
#endif 
/*    std::cerr<<"cudaLaunchKernel: ptr2CUfunc_param size: "
	    <<ptr2CUfunc_param->size()<<" ,ptr: "<<&ptr2CUfunc_param<<std::endl;
    if (ptr2CUfunc_param->size() == 0){
	    std::cerr<<"SIZEE 000000 !!"<<std::endl;
	    abort();
    }*/
    CUfunction kernelFunc;
    int argsNum;
#ifdef DEBUG
    try {
#endif
	// Get the pair associated with the key
	const auto& pair = ptr2CUfunc_param->at(func);
    	// Access the kernel name in the tupple
	kernelFunc = std::get<0>(pair);
    	// Access the number of args
	argsNum = std::get<1>(pair);
#ifdef DEBUG
    } catch (std::out_of_range& e) {
        std::cout << "Error: " << e.what() << ". Key " << func<<" found in the map." << std::endl;
        abort();
    } 
#endif

#ifdef TIMERS 
    e_index_map1 = rdtsc();
    //e_map_chrono = std::chrono::high_resolution_clock::now();
    s_args = rdtsc();
#endif    

    //printf("Intercepted launch of kernel %p with grid size %d x %d x %d and block size %d x %d x %d - NEW KRNL: %p\n", func, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, kernelFunc);

    /*long int deviceExtraArgument = 0xfffffffffffffff;
    void **newArgs = (void **)malloc(sizeof(void *) * (argsNum + 1));
    memcpy(newArgs, args, sizeof(void *) * argsNum);
    newArgs[argsNum] = &deviceExtraArgument;*/

#ifdef TIMERS
    e_args = rdtsc();
    s_compute = rdtsc();
#endif

    CUresult err = cuLaunchKernel(kernelFunc, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, sharedMem, stream, args, NULL);

#ifdef TIMERS
    CUDA_ERROR_FATAL(cuCtxSynchronize());
    e_compute = rdtsc();
#endif 
   if (err == CUDA_SUCCESS)
        return cudaSuccess;
    else
	return cudaErrorUnknown;
// NO NEW PTX //
#else
    std::cerr<<"Native cudaLaunchKernel!!! "<<std::endl;
    printf("Intercepted launch of kernel %p with grid size %d x %d x %d and block size %d x %d x %d\n", func, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);


    orig_cudaLaunchKernel = (cudaError_t (*)(const void*, dim3, dim3, void**, size_t, cudaStream_t))dlsym(RTLD_NEXT,"cudaLaunchKernel");
#ifdef TIMERS
    s_compute = rdtsc();
#endif

    CUDA_ERROR_FATAL_RUNTIME(orig_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream));

#ifdef TIMERS
    CUDA_ERROR_FATAL_RUNTIME(cudaDeviceSynchronize());
    e_compute = rdtsc();
#endif

return cudaSuccess;
#endif
}

/*cudaError_t (*cudaFree_original) (void *);
cudaError_t cudaFree(void *devPtr) {
    std::cerr<<"Intercepted "<<__func__<<std::endl;
    if (!cudaFree_original) {
        cudaFree_original = (cudaError_t (*)(void *)) dlsym(RTLD_NEXT, "cudaFree");
    }
    // Call the original cudaFree function
    return cudaFree_original(devPtr);
}


cudaError_t (*cudaDeviceReset_original) ();
cudaError_t cudaDeviceReset() {
    std::cerr << "Intercept cudaDeviceReset" << std::endl;
    if (!cudaDeviceReset_original) {
        cudaDeviceReset_original = (cudaError_t (*)()) dlsym(RTLD_NEXT, "cudaDeviceReset");
    }
#ifdef TIMERS
    std::cerr<<"====== Inside cudaLaunchKrnl ======"<<std::endl;
    printf("Index cycles: %.3lu \n", (e_index_map1 - s_index_map1));
    printf("Args cycles: %.3lu \n", (e_args - s_args));
    printf("PTX cycles: %.3lu \n", (e_compute - s_compute));
    
    //std::chrono::duration<double, std::milli> map_milli = e_map_chrono - s_map_chrono;
    //std::cerr << "Map time : " << map_milli.count() << " ms" << std::endl;
    //e_compute = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double, std::milli> compute_milli = e_compute - s_compute;
    //std::cerr << "PTX time : " << compute_milli.count() << " ms" << std::endl;
    //std::chrono::duration<double, std::milli> register_milli = e_1 - s_1;
    //std::cerr << "RegisterFunc time : " << register_milli.count() << " ms" << std::endl;
    std::cerr<<"===================================="<<std::endl;                                                                                  
#endif
 
   // Call the original cudaFree function
    return cudaDeviceReset_original();
}*/
/********************** NEW FUNCTIONS *******************************/
// Define function pointers for original functions
#if 0
CUresult (*cuDeviceGetAttribute_original) (int *, CUdevice_attribute, CUdevice);
CUresult (*cuDeviceGetName_original) (char *, int, CUdevice);
CUresult (*cuDeviceGetPCIBusId_original) (char *, int, CUdevice);
CUresult (*cuDeviceGet_original) (CUdevice *, int);
CUresult (*cuDeviceGetCount_original) (int *);
CUresult (*cuDeviceTotalMem_original) (size_t *, CUdevice);
CUresult (*cuDeviceGetUuid_original) (CUuuid *, CUdevice);
cudaError_t (*cudaMalloc_original) (void **devPtr, size_t size);
cudaError_t (*cudaDeviceGetAttribute_original) (int *, cudaDeviceAttr, int);
cudaError_t (*cudaEventCreateWithFlags_original) (cudaEvent_t *, unsigned int);
cudaError_t (*cudaFuncSetAttribute_original) (const void *, cudaFuncAttribute, int);
cudaError_t (*cudaGetDevice_original) (int *device);
cudaError_t (*cudaMemcpy_original) (void *dst, const void *src, size_t count, cudaMemcpyKind kind);

cudaError_t cudaGetDevice(int *device) {
    std::cerr<<"Intercepted "<<__func__<<std::endl;
    if (!cudaGetDevice_original) {
        cudaGetDevice_original = (cudaError_t (*)(int *)) dlsym(RTLD_NEXT, "cudaGetDevice");
        if (!cudaGetDevice_original) {
            fprintf(stderr, "Error: Failed to find original cudaGetDevice function: %s\n", dlerror());
            return cudaErrorUnknown;
        }
    }
    // Call the original cudaGetDevice function
    cudaError_t result = cudaGetDevice_original(device);
    if (result != cudaSuccess) {
        fprintf(stderr, "Error: cudaGetDevice returned %s\n", cudaGetErrorString(result));
    }
    return result;
}

cudaError_t cudaFuncSetAttribute(const void *func, cudaFuncAttribute attr, int value) {
    std::cerr<<"Intercepted "<<__func__<<" func: "<<func<<" name: "<<ptr2name[func]<<std::endl;
    //std::cerr<<"Intercepted "<<__func__<<std::endl;
    if (!cudaFuncSetAttribute_original) {
        cudaFuncSetAttribute_original = (cudaError_t (*)(const void *, cudaFuncAttribute, int)) dlsym(RTLD_NEXT, "cudaFuncSetAttribute");
        if (!cudaFuncSetAttribute_original) {
            fprintf(stderr, "Error: Failed to find original cudaFuncSetAttribute function: %s\n", dlerror());
            exit(EXIT_FAILURE);
        }
    }
    // Call the original cudaFuncSetAttribute function
    cudaError_t result = cudaFuncSetAttribute_original(func, attr, value);
    if (result != cudaSuccess) {
        fprintf(stderr, "Error: cudaFuncSetAttribute returned %d\n", result);
    }
    return result;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) {
    std::cerr<<"Intercepted "<<__func__<<std::endl;
    if (!cudaEventCreateWithFlags_original) {
        cudaEventCreateWithFlags_original = (cudaError_t (*)(cudaEvent_t *, unsigned int)) dlsym(RTLD_NEXT, "cudaEventCreateWithFlags");
        if (!cudaEventCreateWithFlags_original) {
            fprintf(stderr, "Error: Failed to find original cudaEventCreateWithFlags function: %s\n", dlerror());
            exit(EXIT_FAILURE);
        }
    }
    // Call the original cudaEventCreateWithFlags function
    cudaError_t result = cudaEventCreateWithFlags_original(event, flags);
    if (result != cudaSuccess) {
        fprintf(stderr, "Error: cudaEventCreateWithFlags returned %d\n", result);
    }
    return result;
}

cudaError_t cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int device) {
    std::cerr<<"Intercepted "<<__func__<<std::endl;
    if (!cudaDeviceGetAttribute_original) {
        cudaDeviceGetAttribute_original = (cudaError_t (*)(int *, cudaDeviceAttr, int)) dlsym(RTLD_NEXT, "cudaDeviceGetAttribute");
        if (!cudaDeviceGetAttribute_original) {
            fprintf(stderr, "Error: Failed to find original cudaDeviceGetAttribute function: %s\n", dlerror());
            exit(EXIT_FAILURE);
        }
    }
    // Call the original cudaDeviceGetAttribute function
    cudaError_t result = cudaDeviceGetAttribute_original(value, attr, device);
    if (result != cudaSuccess) {
        fprintf(stderr, "Error: cudaDeviceGetAttribute returned %d\n", result);
    }
    return result;
}

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    std::cerr<<"Intercepted "<<__func__<<std::endl;
    if (!cudaMalloc_original) {
        cudaMalloc_original = (cudaError_t (*)(void **, size_t)) dlsym(RTLD_NEXT, "cudaMalloc");
        if (!cudaMalloc_original) {
            fprintf(stderr, "Error: Failed to find original cudaMalloc function: %s\n", dlerror());
            return cudaErrorUnknown;
        }
    }
    // Call the original cudaMalloc function
    cudaError_t result = cudaMalloc_original(devPtr, size);
    if (result != cudaSuccess) {
        fprintf(stderr, "Error: cudaMalloc returned %s\n", cudaGetErrorString(result));
    }
    return result;
}
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) {
    std::cerr<<"Intercepted "<<__func__<<std::endl;
    if (!cudaMemcpy_original) {
        cudaMemcpy_original = (cudaError_t (*)(void *, const void *, size_t, cudaMemcpyKind)) dlsym(RTLD_NEXT, "cudaMemcpy");
        if (!cudaMemcpy_original) {
            fprintf(stderr, "Error: Failed to find original cudaMemcpy function: %s\n", dlerror());
            return cudaErrorUnknown;
        }
    }
    // Call the original cudaMemcpy function
    cudaError_t result = cudaMemcpy_original(dst, src, count, kind);
    if (result != cudaSuccess) {
        fprintf(stderr, "Error: cudaMemcpy returned %s\n", cudaGetErrorString(result));
    }
    return result;
}

CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
    std::cerr<<"Intercepted "<<__func__<<std::endl;
    if (!cuDeviceGetAttribute_original) {
        cuDeviceGetAttribute_original = (CUresult (*)(int *, CUdevice_attribute, CUdevice)) dlsym(RTLD_NEXT, "cuDeviceGetAttribute");
        if (!cuDeviceGetAttribute_original) {
            fprintf(stderr, "Error: Failed to find original cuDeviceGetAttribute function: %s\n", dlerror());
            exit(EXIT_FAILURE);
        }
    }
    // Call the original cuDeviceGetAttribute function
    CUresult result = cuDeviceGetAttribute_original(pi, attrib, dev);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "Error: cuDeviceGetAttribute returned %d\n", result);
    }
    return result;
}

CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    std::cerr<<"Intercepted "<<__func__<<std::endl;
    if (!cuDeviceGetName_original) {
        cuDeviceGetName_original = (CUresult (*)(char *, int, CUdevice)) dlsym(RTLD_NEXT, "cuDeviceGetName");
        if (!cuDeviceGetName_original) {
            fprintf(stderr, "Error: Failed to find original cuDeviceGetName function: %s\n", dlerror());
            exit(EXIT_FAILURE);
        }
    }
    // Call the original cuDeviceGetName function
    CUresult result = cuDeviceGetName_original(name, len, dev);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "Error: cuDeviceGetName returned %d\n", result);
    }
    return result;
}

CUresult cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
    std::cerr<<"Intercepted "<<__func__<<std::endl;
    if (!cuDeviceGetPCIBusId_original) {
        cuDeviceGetPCIBusId_original = (CUresult (*)(char *, int, CUdevice)) dlsym(RTLD_NEXT, "cuDeviceGetPCIBusId");
        if (!cuDeviceGetPCIBusId_original) {
            fprintf(stderr, "Error: Failed to find original cuDeviceGetPCIBusId function: %s\n", dlerror());
            return CUDA_ERROR_INVALID_VALUE;
        }
    }
    // Call the original cuDeviceGetPCIBusId function
    CUresult result = cuDeviceGetPCIBusId_original(pciBusId, len, dev);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "Error: cuDeviceGetPCIBusId returned %d\n", result);
    }
    return result;
}

CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    std::cerr<<"Intercepted "<<__func__<<std::endl;
    if (!cuDeviceGet_original) {
        cuDeviceGet_original = (CUresult (*)(CUdevice *, int)) dlsym(RTLD_NEXT, "cuDeviceGet");
        if (!cuDeviceGet_original) {
            fprintf(stderr, "Error: Failed to find original cuDeviceGet function: %s\n", dlerror());
            exit(EXIT_FAILURE);
        }
    }
    // Call the original cuDeviceGet function
    CUresult result = cuDeviceGet_original(device, ordinal);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "Error: cuDeviceGet returned %d\n", result);
    }
    return result;
}

CUresult cuDeviceGetCount(int *count) {
    std::cerr<<"Intercepted "<<__func__<<std::endl;
    if (!cuDeviceGetCount_original) {
        cuDeviceGetCount_original = (CUresult (*)(int *)) dlsym(RTLD_NEXT, "cuDeviceGetCount");
        if (!cuDeviceGetCount_original) {
            fprintf(stderr, "Error: Failed to find original cuDeviceGetCount function: %s\n", dlerror());
            exit(EXIT_FAILURE);
        }
    }
    // Call the original cuDeviceGetCount function
    CUresult result = cuDeviceGetCount_original(count);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "Error: cuDeviceGetCount returned %d\n", result);
    }
    return result;
}

CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
    std::cerr<<"Intercepted "<<__func__<<std::endl;
    if (!cuDeviceTotalMem_original) {
        cuDeviceTotalMem_original = (CUresult (*)(size_t *, CUdevice)) dlsym(RTLD_NEXT, "cuDeviceTotalMem");
        if (!cuDeviceTotalMem_original) {
            fprintf(stderr, "Error: Failed to find original cuDeviceTotalMem function: %s\n", dlerror());
            exit(EXIT_FAILURE);
        }
    }
    // Call the original cuDeviceTotalMem function
    CUresult result = cuDeviceTotalMem_original(bytes, dev);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "Error: cuDeviceTotalMem returned %d\n", result);
    }
    return result;
}
CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
    std::cerr<<"Intercepted "<<__func__<<std::endl;
    if (!cuDeviceGetUuid_original) {
        cuDeviceGetUuid_original = (CUresult (*)(CUuuid *, CUdevice)) dlsym(RTLD_NEXT, "cuDeviceGetUuid");
        if (!cuDeviceGetUuid_original) {
            fprintf(stderr, "Error: Failed to find original cuDeviceGetUuid function: %s\n", dlerror());
            return CUDA_ERROR_INVALID_VALUE;
        }
    }
    // Call the original cuDeviceGetUuid function
    CUresult result = cuDeviceGetUuid_original(uuid, dev);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "Error: cuDeviceGetUuid returned %d\n", result);
    }
    return result;
}
#endif
}
