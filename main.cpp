//Inner product (dot product) of two vectors in a parallel fashion
#include <stdlib.h> 
#include <math.h>

#define N (pow(2,14))

void cuda_function(float *d_a, float *d_b, float *d_c, float *a, float *b, float *c, int NN);

int main(void)
{

  float *a, *b, *c;// host copies of a, b, c
  float *d_a, *d_b, *d_c;// device copies of a, b, c
  float size = N * sizeof(float);

  a = (float *)malloc(size);
  b = (float *)malloc(size);
  c = (float *)malloc(sizeof(float));

  for (int i = 0; i < N; i++) {
    a[i] = 1;
    b[i] = 1;
  }

  cuda_function(d_a, d_b, d_c, a, b, c, N);

  // Remember: free and cudaFree DO NOT ERASE MEMORY! They only
  // return memory to a pool to be re-allocated. That is why the shared
  // variable 'cc' is initialized inside the kernel. See this:
  // http://stackoverflow.com/questions/13100615/cudafree-is-not-freeing-memory
  free(a);
  free(b);
  free(c);

  return 0;
}
