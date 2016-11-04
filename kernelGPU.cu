#include <cuda.h>
#include <stdio.h>
#include <stdlib.h> 
#include <iostream>

#define THREADS_PER_BLOCK 32

__global__ void innerProd(float *aa, float *bb, float *cc)
{
   __shared__ float temp[THREADS_PER_BLOCK];
   int index = threadIdx.x + blockIdx.x* blockDim.x;
   temp[threadIdx.x] = aa[index]*bb[index];

   *cc = 8; // Initialized to avoid memory problems. See comments
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

void cuda_function(float *ainput, float *binput, float *cinput, int NN)
{
  std::cout << ">>> inside cuda_function " << std::endl;
  std::cout << "NN = " << NN << "\n";
  
  #define NUMBER_OF_BLOCKS (NN/THREADS_PER_BLOCK)
  float *d_ainput, *d_binput, *d_cinput;// device copies of a, b, c
  //float GPU_profile;
  float size = NN * sizeof(float);

  ainput = (float *)malloc(size);
  binput = (float *)malloc(size);
  cinput = (float *)malloc(sizeof(float));
  *cinput = 7;

   for (int i = 0; i < NN; i++) {
    ainput[i] = 1;
    binput[i] = 1;
  }

  std::cout << "a[0] = " << ainput[0] << "\n";
  std::cout << "b[0] = " << binput[0] << "\n";



float test = 67.0f;

int retMem = 99;
int retMalloc = 99;


  //std::cout << "size = " << size << "\n";

  // ----- Variables to profile the execution time
  //cudaEvent_t start, stop;
  //cudaEventCreate(&start);
  //cudaEventCreate(&stop);
	
  // In the GPU ------------------------------------------
  retMalloc = cudaMalloc((void**)&*d_ainput, 1024*4);
  cudaMalloc((void**)&d_binput, size);
  cudaMalloc((void**)&d_cinput, sizeof(float));

  cudaMemcpy(d_ainput, ainput, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_binput, binput, size, cudaMemcpyHostToDevice);
  retMem = cudaMemcpy(d_cinput, cinput, sizeof(float), cudaMemcpyHostToDevice);

  // Call kernel
  //cudaEventRecord(start);
  innerProd<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(d_ainput, d_binput, d_cinput);
  //cudaEventRecord(stop);

  cudaMemcpy(cinput, d_cinput, sizeof(float), cudaMemcpyDeviceToHost);
  //cudaEventSynchronize(stop);

  // Elapsed time -- GPU
  //cudaEventElapsedTime(&GPU_profile, start, stop);
  // -----------------------------------------------------

  std::cout << "retMalloc = " << retMalloc << "\n"; 
  std::cout << "retMem = " << retMem << "\n";  
std::cout << "NUMBER_OF_BLOCKS = " << NUMBER_OF_BLOCKS << "\n";
  std::cout << "c = " << *cinput << "\n";
 // std::cout << "Kernel execution time in GPU = " << GPU_profile <<
   //     " milliseconds" << "\n";


    std::cout << ">>> Free memory " << "\n";
   // Remember: free and cudaFree DO NOT ERASE MEMORY! They only
  // return memory to a pool to be re-allocated. That is why the shared
  // variable 'cc' is initialized inside the kernel. See this:
  // http://stackoverflow.com/questions/13100615/cudafree-is-not-freeing-memory
  //free(ainput);
  //free(binput);
  //free(cinput);

  cudaFree(d_ainput);
  cudaFree(d_binput);
  cudaFree(d_cinput);
}
