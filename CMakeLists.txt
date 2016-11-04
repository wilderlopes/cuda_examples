# CMakeLists.txt to build application_profiling.cu
cmake_minimum_required(VERSION 2.8)
find_package(CUDA REQUIRED)

# Specify binary name and source file to build it from
cuda_add_executable(
    main
    main.cpp
    kernelGPU.cu)
