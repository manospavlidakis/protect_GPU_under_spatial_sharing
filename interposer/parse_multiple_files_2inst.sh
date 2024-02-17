#!/bin/bash
for mf in `find -name '*ptx'`; do
	echo $mf
	./../../bound_checking_paper/ptx_parser/parse_ptx_v2.py $mf ../mod_caffe_libs/$mf 
done
