#!/bin/bash
for mf in `find -name '*ptx'`; do
	echo "Compiling " $mf
	nvcc -fatbin $mf -arch=sm_80
	if [ $? -ne 0 ]; then
		echo "An error occurred"
		exit
	fi
done
