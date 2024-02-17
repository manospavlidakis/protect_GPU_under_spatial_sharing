#!/bin/bash
g++ main.cpp -o app -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -lcudart
