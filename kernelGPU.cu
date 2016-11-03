#include <cuda.h>
#include <stdio.h>
#include <iostream>

#define THREADS_PER_BLOCK 512

__global__ void innerProd(float *aa, float *bb, float *cc)
{
   __shared__ float temp[THREADS_PER_BLOCK];
   int index = threadIdx.x + blockIdx.x* blockDim.x;
   temp[threadIdx.x] = aa[index]*bb[index];

   *cc = 0; // Initialized to avoid memory problems. See comments
            // below, next to the free and cudaFree commands.

   // No thread goes beyond this point until all of them
   // have reached it. Threads are only synchronized within
   // a block.
   __syncthreads();

   //  Thread 0 sums the pairwise products
   if (threadIdx.x == 0) {
     float sum = 0;
     for (int i = 0; i < THREADS_PER_BLOCK; i++){
       sum += temp[i];
     }
      // Use atomicAdd to avoid different blocks accessing cc at the
      // same time (race condition). The atomic opperation enables
      // read-modify-write to be performed by a block without interruption.
      //*cc += sum;
    atomicAdd(cc, sum);
   }

}


void cuda_function(float *d_a, float *d_b, float *d_c, float *a, float *b, float *c, int NN)
{

  #define NUMBER_OF_BLOCKS (NN/THREADS_PER_BLOCK)
  float GPU_profile;
  float size = NN * sizeof(float);

  // ----- Variables to profile the execution time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
	
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
  std::cout << "Kernel execution time in GPU = " << GPU_profile <<
        " milliseconds" << "\n";

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
