#include <stdio.h>
#include <stdlib.h>

__global__ void add(int *a, int *b, int *c) {
c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

void random_ints(int* x, int size)
{
	int i;
	for (i=0;i<size;i++) {
		x[i]=rand()%50;
	}
}
#define N 512
    int main(void) {
int *a, *b, *c;		// host copies of a, b, c
int *d_a, *d_b, *d_c;	//device copies
int tempCounter =0;
	int size = N * sizeof(int);
// Alloc space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
// Alloc space for host copies and initialize 
	a = (int *)malloc(size); random_ints(a, N);
	b = (int *)malloc(size); random_ints(b, N);
	c = (int *)malloc(size);

// Copy inputs to device
cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

// Launch add() kernel on GPU with N blocks
       add<<<N,1>>>(d_a, d_b, d_c);
// Copy result back to host
cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

       for(tempCounter =0; tempCounter < N; tempCounter++)
       {
        printf("%d + %d  = %d \n", a[tempCounter], b[tempCounter], c[tempCounter]); 
       }
// Cleanup
        free(a); free(b); free(c);
cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

