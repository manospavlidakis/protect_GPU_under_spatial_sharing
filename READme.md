# Guardian(G-Safe) components: To protect GPU memory under sharing 

## cu_extract
- extracts PTX from a .so/.a file and generates klist.json.
- compile using build.sh.
- it requires arax.
- for pytorch: find libtorch_cuda.so (https://carvgit.ics.forth.gr/manospavl/pytorch_static)


## ptx_parser
- modifies PTX (add extra parameter, extra register, load parameter to register, and bitmasking).
- If you use parse_multiple_files.sh it performs cuobjdump and copies the modified ptxs to a dir called modified_cublas_ptx.

## interposer
- intercepts cudaRegister and cudaLaunchKernel.
- 3 version: native (only interception), if (call ptx with base+final partition addr), 2instrcutions(callptx with base+mask)
- cudaRegister uses the klist to create a mapping with kernel symbol name as the key and ptx, #param as values. Additionally, it generates a map with hostFun and deviceName (ptr2name). The hostFun is the ptr of a kernel and the deviceName is the symbol name of the kernel. 
- cudaLaunchKernel finds the symbol name of the kernel (fname) using the ptr2name map. Then it uses the fname as key to find the ptx and the #parameters (from kernelMap). Finally, it call the modified kernel from the ptx generated from the ptx_parser.
