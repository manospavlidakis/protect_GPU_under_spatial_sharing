#!/bin/bash
# copy this file to the directory with the native ptx and adjust the path

for mf in `find -name '*ptx'`; do
	echo $mf
	/home1/public/manospavl/bound_checking_paper/ptx_parser/parse_ptx_guardian.py $mf ../ptx_pytorch_mod_full/$mf
done
