# PTX patcher
1. parse_ptx_guardian.py: adds two bitwise instructions before every load/store instruction. It uses a mask and the base partition address.
2. parse_ptx_if.py: add two branch instructions before every load/store instruction. It uses the partition base address and ending address.

## Run parse_ptx_guardian.py <original ptx> <guardian ptx>
## Test directory contains simple kernels 
