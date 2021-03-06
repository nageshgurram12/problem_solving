#include <stdio.h>

#define SIZE 256

__global__ void staticReverse(int *d, int n)
{
  int s[SIZE]; //Here s is allocated in local memory, 
  int t = threadIdx.x;
  int tr = n-t-1;
  s[t] = d[t];
//  __syncthreads();
  d[t] = s[tr];
}

int main(void)
{
  const int n = SIZE;
  int a[n], r[n], d[n];
  
  for (int i = 0; i < n; i++) {
    a[i] = i;
    r[i] = n-i-1;
    d[i] = 0;
  }

  int *d_d;
  cudaMalloc(&d_d, n * sizeof(int)); 
  
  // run version with static shared memory
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  staticReverse<<<1,n>>>(d_d, n);
  cudaMemcpy(d, d_d, n*sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) 
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);
  
}
