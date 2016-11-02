//Inner product (dot product) of two vectors in a parallel fashion
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#define N 1024
#define THREADS_PER_BLOCK 512
#define NUMBER_OF_BLOCKS (N/THREADS_PER_BLOCK)


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

int main(void)
{
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
    a[i] = 2;
    b[i] = 0.5;
  }

  cudaMalloc((void**)&d_a, size);
  cudaMalloc((void**)&d_b, size);
  cudaMalloc((void**)&d_c, sizeof(float));

  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  // Call kernel
  innerProd<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
  // innerProd<<<1, N>>>(d_a, d_b, d_c);

  cudaMemcpy(c, d_c, sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "c = " << *c << "\n";

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
