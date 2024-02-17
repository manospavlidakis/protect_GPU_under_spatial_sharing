#!/bin/bash
g++ cu_extract.cpp -I ../arax/build/include/ -I ../arax/controller/include/ ../arax/build/libarax_st.a -lrt -lpthread

g++ cu_extract_dir.cpp -I ../arax/build/include/ -I ../arax/controller/include/ ../arax/build/libarax_st.a -lrt -lpthread
