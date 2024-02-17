#!/bin/bash
#### sudo yum install jq -y
rm -rf rodinia_ptx/
rm -rf cu_*ptx
mkdir rodinia_ptx
for path in ./rodinia_libs/*
do
	echo $path
	file=$(basename "$path" .so)
	echo $file
	./a.out $path ${file}_ptx
	# remove empty keys
	jq 'del(.[""])' ${file}_ptx/klist.json > ${file}_ptx/klist_mod.json 
	# remove the first line containing {
	sed -i '1d' ${file}_ptx/klist_mod.json 
        # remove 75:{
	sed -i 's/"75": {//g' ${file}_ptx/klist_mod.json
	# remove the last line which contains }
	sed -i '$d' ${file}_ptx/klist_mod.json
	mv ${file}_ptx/*ptx rodinia_ptx/
done
# merge all klists
cat cu_bfs_ptx/klist_mod.json cu_gaussian_ptx/klist_mod.json cu_hotspot3D_ptx/klist_mod.json cu_hotspot_ptx/klist_mod.json cu_lavamd_ptx/klist_mod.json cu_nn_ptx/klist_mod.json cu_nw_ptx/klist_mod.json cu_particle_float_ptx/klist_mod.json cu_pathfinder_ptx/klist_mod.json > klist_all_rodinia.json

# add sm 80 to the beggining of the NEW json
sed -i '1s/^/{ "75": {\n/' klist_all_rodinia.json
sed -i '$a\}' klist_all_rodinia.json

#jq -s '.' cu_*/*.json > klist_all_rodinia.json
echo "Json validity!!!!"
jq empty klist_all_rodinia.json  && echo "Valid JSON" || echo "Invalid JSON"
#jq -s '.' cu_bfs_ptx/klist.json cu_gaussian_ptx/klist.json cu_hotspot3D_ptx/klist.json cu_hotspot_ptx/klist.json cu_lavamd_ptx/klist.json cu_nn_ptx/klist.json cu_nw_ptx/klist.json cu_particle_float_ptx/klist.json cu_pathfinder_ptx/klist.json > klist_all_rodinia.json

#rm -rf cu_*_ptx
