//Sum of two vectors in a parallel fashion 

#include <stdio.h>
#include <iostream>
#define N 512

__global__
void add(int *aa, int *bb, int *cc)
{
   //int index = threadIdx.x + blockIdx.x* blockDim.x;
//   int index = threadIdx.x;
   int index = blockIdx.x;
   cc[index] = aa[index] + bb[index];  
}

int main(void)
{
  
  int *a, *b, *c;// host copies of a, b, c
  int *d_a, *d_b, *d_c;// device copies of a, b, c
  int size = N * sizeof(int);

  a = (int *)malloc(size); 
  b = (int *)malloc(size); 
  c = (int *)malloc(size);

  for (int i = 0; i < N; i++) {
    a[i] = 1;
    b[i] = 2;
    c[i] = -1;
  }

  cudaMalloc((void**)&d_a, size);
  cudaMalloc((void**)&d_b, size); 
  cudaMalloc((void**)&d_c, size);
  
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  add<<<N, 1>>>(d_a, d_b, d_c);

  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  //free(a); 
  //free(b); 
  //free(c);
  cudaFree(d_a); 
  cudaFree(d_b); 
  cudaFree(d_c);
  //return 0;
  //std::cout << "Result c\n";
  std::cout << "a = " << a[1] << "\n";
  std::cout << "b = " << b[1] << "\n";
  std::cout << "c = " << c[1] << "\n";


}
