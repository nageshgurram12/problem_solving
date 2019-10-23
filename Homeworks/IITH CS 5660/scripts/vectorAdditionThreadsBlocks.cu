#include <stdio.h>
#include <stdlib.h>
//#define N (2048*2048)
#define N (2*2048)
#define THREADS_PER_BLOCK 512


__global__ void add(int *a, int *b, int *c)
 {
int index = threadIdx.x + blockIdx.x * blockDim.x;
//    printf(" index  = %d\n ", index);
    c[index] = a[index] + b[index];
}

void random_ints(int* x, int size)
{
        int i;
        for (i=0;i<size;i++) {
                x[i]=rand()%50;
        }
}

    int main(void) {
 int tempCounter =0;
 int *a, *b, *c; // host copies of a, b, c
 int *d_a, *d_b, *d_c; // device copies            
 int size = N * sizeof(int);
  // Alloc space for device copies of a, b, c
        cudaMalloc((void **)&d_a, size);
        cudaMalloc((void **)&d_b, size);
        cudaMalloc((void **)&d_c, size);
a = (int *)malloc(size); random_ints(a, N);
b = (int *)malloc(size); random_ints(b, N);
c = (int *)malloc(size);

 // Copy inputs to device
cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
        // Launch add() kernel on GPU
add<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
        // Copy result back to host
cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
       // Cleanup
       for(tempCounter =0; tempCounter < N; tempCounter++)
       {
        printf("%d + %d  = %d \n", a[tempCounter], b[tempCounter], c[tempCounter]);
       }
      
        free(a); free(b); free(c);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

