//Submission should be named as  <RollNumber>_Prog.cu
//Upload just this cu file and nothing else. If you upload it as a zip, it will not be evaluated. 


#include <stdio.h>
#define M 514 
//Input has 514 rows and columns 

#define N 512 
//For output, only 512 rows and columns need to be computed. 


//TODO: WRITE GPU KERNEL. It should not be called repeatedly from the host, but just once. Each time it is called, it may process more than pixel or not process any pixel at all. 

main (int argc, char **argv) {
  int A[M][M], B[M][M];
  int *d_A, *d_B; // These are the copies of A and B on the GPU
  int *h_B;       // This is a host copy of the output of B from the GPU
  int i, j;

  // Input is randomly generated
  for(i=0;i<M;i++) {
    for(j=0;j<M;j++) {
      A[i][j] = rand()/1795831;
      //printf("%d\n",A[i][j]);
    }
  }

  // sequential implementation of main computation
  for(i=1;i<M-1;i++) {
    for(j=1;j<M-1;j++) {
      B[i][j] = (A[i-1][j]+A[i+1][j]+A[i][j-1]+A[i][j+1])/4;
    }
  }


  // TODO: ALLOCATE MEMORY FOR GPU COPIES OF d_A AND d_B

  // TODO: COPY A TO d_A

  // TODO: CREATE BLOCKS with THREADS AND INVOKE GPU KERNEL
   //Use 9 blocks, each with 48 threads

  // TODO: COPY d_B BACK FROM GPU to CPU in variable h_B

  // TODO: Verify result is correct by comparing
  for(i=1;i<M-1;i++) {
    for(j=1;j<M-1;j++) {
    //TODO: compare each element of h_B and B by subtracting them
        //print only those elements for which the above subtraction is non-zero
    }
   }
    //IF even one element of h_B and B differ, report an error.
    //Otherwise, there is no error.
    //If your program is correct, no error should occur.
}

/*Remember the following guidelines to avoid losing marks
Index of an array should not exceed the array size. 
Do not ignore the fact that boundary rows and columns need not be computed (in fact, they cannot be computed since they don't have four neighbors)
No output array-element should be computed more than once
No marks will be given if the program does not compile or run (TAs will not debug your program at all)
*/
