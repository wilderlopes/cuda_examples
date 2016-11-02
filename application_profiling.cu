//Inner product (dot product) of two vectors in a parallel fashion
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <time.h>
#include <math.h>

#define N (pow(2,14))
#include "kernelsGPU.cuh"
#include "kernelsCPU.h"



int main(void)
{
  // ----- Variables to profile the execution time
  float CPU_profile, GPU_profile;
  // CPU
  clock_t startCPU, endCPU;
  // GPU
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  //----------------------------------------------

  float *a, *b, *c;// host copies of a, b, c
  float *d_a, *d_b, *d_c;// device copies of a, b, c
  float size = N * sizeof(float);
  //int sizeInGPU;

  a = (float *)malloc(size);
  b = (float *)malloc(size);
  c = (float *)malloc(sizeof(float));

  // Define QoS: p0
  // supervisor(float *lambda_GPU)
  // sizeInGPU = lambda_GPU*N;

  for (int i = 0; i < N; i++) {
    a[i] = 1;
    b[i] = 1;
  }

  // In the CPU ------------------------------------------
  startCPU = clock();
  innerProdCPU(a, b, c);
  endCPU = clock();
  // Elapsed time -- CPU 
  CPU_profile = (((double) (endCPU - startCPU)) / CLOCKS_PER_SEC)*1000;
  // -----------------------------------------------------

  // In the GPU ------------------------------------------
  cudaMalloc((void**)&d_a, size);
  cudaMalloc((void**)&d_b, size);
  cudaMalloc((void**)&d_c, sizeof(float));

  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  // Call kernel
  cudaEventRecord(start);
  innerProd<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
  cudaEventRecord(stop);

  cudaMemcpy(c, d_c, sizeof(float), cudaMemcpyDeviceToHost);
  cudaEventSynchronize(stop);

  // Elapsed time -- GPU
  cudaEventElapsedTime(&GPU_profile, start, stop);
  // -----------------------------------------------------

  std::cout << "NUMBER_OF_BLOCKS = " << NUMBER_OF_BLOCKS << "\n";
  std::cout << "c = " << *c << "\n";
  std::cout << "Kernel execution time in CPU = " << CPU_profile <<
        " milliseconds" << "\n";
  std::cout << "Kernel execution time in GPU = " << GPU_profile <<
        " milliseconds" << "\n";


  // Remember: free and cudaFree DO NOT ERASE MEMORY! They only
  // return memory to a pool to be re-allocated. That is why the shared
  // variable 'cc' is initialized inside the kernel. See this:
  // http://stackoverflow.com/questions/13100615/cudafree-is-not-freeing-memory
  free(a);
  free(b);
  free(c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
