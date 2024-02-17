#!/bin/bash

# Set the directory containing the files to rename
dir="modified_cusparse"

# Set the prefix for the new file names
prefix="libcusparse_static."

# Loop through all files in the directory
for file in "$dir"/libcusparse.*.ptx
do
    echo "file" ${file}
    # Get the file extension
    extension="${file##*.}"
    number=$(echo "$file" | sed 's/.*\.\([0-9]\+\)\.sm_.*/\1/')
    echo "number: "$number

    # Get the new name
    new_name="$prefix$number.sm_86.$extension"
    echo "new: "$new_name

    # Rename the file with the new name
    mv "$file" "$dir/$new_name"
done
