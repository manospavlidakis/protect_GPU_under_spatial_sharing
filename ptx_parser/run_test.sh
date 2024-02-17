#!/bin/bash
echo "Generate modified ptx"
./parse_ptx.py test/empty_kernel.ptx test/my.ptx
echo "======================="
echo "Run app with modified ptx"
./test/app test/my.ptx 
echo "======================="
echo "Run app with ORIGINAL pts"
./test/app test/empty_kernel.ptx
