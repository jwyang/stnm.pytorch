#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

cd src
echo "Compiling stnm kernels by nvcc..."
nvcc -c -o stnm_cuda_kernel.cu.o stnm_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../
python build.py
