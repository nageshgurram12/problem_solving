#include <stdio.h>
#include <math.h>
#define M 514
//Input has 514 rows and columns

#define N 512
//For output, only 512 rows and columns need to be computed.

#define BLOCKS 9
#define THREADSPERBLOCK 48

// Flattten the 2D array into 1D array
__global__ void image_blur(int* d_A, int* d_B, int pixelsPerThread){
  // Access all pixels for the given thread and calculate blur value for it
  int index = 0;
  int row = 0;
  int  column = 0;
  int i;
  for(i=0; i<pixelsPerThread; i++){
   // get index for that thread 
   index = i + (pixelsPerThread * threadIdx.x) + (blockIdx.x * blockDim.x) * pixelsPerThread;
   row = index % M;
   column = index / M;
   if(row > 0 && row < M-1 && column > 0 && column < M-1){
     d_B[row*M + column] = (d_A[(row-1)*M + column] + d_A[(row+1)*M + column] + d_A[row*M + (column-1)] + d_A[row*M + (column+1)])/4;
   }
  }
}

int main (int argc, char **argv) {
  int A[M][M], B[M][M];
  int *d_A, *d_B; // These are the copies of A and B on the GPU
  int *h_B;       // This is a host copy of the output of B from the GPU
  int i, j;

  // Input is randomly generated
  for(i=0;i<M;i++) {
    for(j=0;j<M;j++) {
      A[i][j] = rand()/1795831;
    }
  }
  // sequential implementation of main computation
  for(i=1;i<M-1;i++) {
    for(j=1;j<M-1;j++) {
      B[i][j] = (A[i-1][j]+A[i+1][j]+A[i][j-1]+A[i][j+1])/4;
    }
  }

  int size = sizeof(int) *  M * M;

  cudaMalloc((void **) & d_A, size);
  cudaMalloc((void **) & d_B, size);

  h_B = (int *) malloc(size);

  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

  int pixelsPerThread = ceil((M * M) / (BLOCKS * THREADSPERBLOCK) + 0.0);
  image_blur<<<BLOCKS, THREADSPERBLOCK>>>(d_A, d_B, pixelsPerThread);
  cudaDeviceSynchronize();

  // TODO: COPY d_B BACK FROM GPU to CPU in variable h_B
  cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

  // TODO: Verify result is correct by comparing
  for(i=1;i<M-1;i++) {
    for(j=1;j<M-1;j++) {
      //print only those elements for which the above subtraction is non-zero
      if(h_B[i*M+j] != B[i][j]){
       printf("!!!! Error at %d row and %d column for host value %d and device value %d !!!\n", i , j, B[i][j], h_B[i*M+j]);
      }
    }
   }

   cudaFree(d_A);
   cudaFree(d_B);
   free(h_B);
}
