//Inner product (dot product) of two vectors in a parallel fashion
#include <stdlib.h>
#include <stdio.h> 
#include <math.h>

#define N (1024)

void cuda_function(float *ainput, float *binput, float *cinput, int NN);

int main(void)
{

  float *a, *b, *c;// host copies of a, b, c
  //float *d_a, *d_b, *d_c;// device copies of a, b, c
  float size = N * sizeof(float);

  //a = (float *)malloc(size);
  //b = (float *)malloc(size);
  //c = (float *)malloc(sizeof(float));
  //d_a = (float *)malloc(size);
  //d_b = (float *)malloc(size);
  //d_c = (float *)malloc(sizeof(float));


  //*c=7.0f;

 
  //std::cout << ">>> before calling cuda_function " << std::endl; 
  cuda_function(a, b, c, N);

 

  return 0;
}
