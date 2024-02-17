# Extractor

## Extract ptx code from executable into provided directory.

1. extract ptx using: cu_extract <executable> <directory>

## The output folder contains:
- klist.json: A JSON file that lists all ptx objects, the contained kernels and the parameters for each kernel.
- ptx files: All files mentioned in the klist.json file.
