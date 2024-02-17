#! /usr/bin/env python3 
import os
import sys

if len(sys.argv) != 2:
    print("Usage: python count_ptx.py <directory_path>")
    sys.exit(1)

directory_path = sys.argv[1]


entry_count = 0
func_count = 0
non_empty_lines = 0
ld_local_count = 0
ld_shared_count = 0
ld_global_count = 0
ld_count = 0
ld_const_count = 0
tex_count = 0
st_local_count = 0
st_shared_count = 0
st_global_count = 0
st_count = 0

for filename in os.listdir(directory_path):
    if filename.endswith(".ptx"):
        ptx_file_path = os.path.join(directory_path, filename)
        with open(ptx_file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            # Count .entry and .func
            if ".entry" in line:
                #print(line)
                entry_count += 1
            elif ".func" in line:
                func_count += 1
        
            # Count non-empty lines
            if line.strip():
                non_empty_lines += 1
            
			# Count ld instructions
            if "ld.local" in line:
                ld_local_count += 1
            elif "ld.shared" in line:
                ld_shared_count += 1
            elif "ld.global" in line:
                ld_global_count += 1
            elif "ld.const" in line:
                ld_const_count += 1
            elif "ld." in line and not "ld.param" in line:
                ld_count += 1
            elif ".tex" in line:
                tex_count += 1
                
            # Count st instructions
            if "st.local" in line:
                st_local_count += 1
            elif "st.shared" in line:
                st_shared_count += 1
            elif "st.global" in line:
                st_global_count += 1
            elif "st." in line:
                st_count += 1

print("Number of .entry statements: {}".format(entry_count))
print("Number of .func statements: {}".format(func_count))
print("Number of non-empty lines: {}".format(non_empty_lines))
print("Number of ld.local instructions: {}".format(ld_local_count))
print("Number of ld.shared instructions: {}".format(ld_shared_count))
print("Number of ld.global instructions: {}".format(ld_global_count))
print("Number of ld.const instructions: {}".format(ld_const_count))
print("Number of ld instructions (excluding ld.local, ld.shared, ld.global, and ld.const): {}".format(ld_count))
print("Number of .tex instructions: {}".format(tex_count))
print("Number of st.local instructions: {}".format(st_local_count))
print("Number of st.shared instructions: {}".format(st_shared_count))
print("Number of st.global instructions: {}".format(st_global_count))
print("Number of st instructions (excluding st.local, st.shared, and st.global): {}".format(st_count))

