#!/bin/bash

# specify the directory path
dir="."

# loop through all files in the directory
for file in "$dir"/*
do
    if [ "${file: -3}" != ".sh" ] && [ "${file: -2}" != ".a" ]; then
	    echo ${file}
	    nvcc -Xptxas="-v" -fatbin ${file} -arch=sm_86 &>> modified_stats.txt
	    if [ "$?" -ne 0 ]; then
		    echo "Compilation failed for ${file}"
		    exit 1
	    fi
    fi
done
