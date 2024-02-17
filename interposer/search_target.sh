#!/bin/bash
for ptx in `find -name '*ptx'`; do
	echo ${ptx}
	sed -i 's/\.target sm_86/\.target sm_80/g' ${ptx}
done
