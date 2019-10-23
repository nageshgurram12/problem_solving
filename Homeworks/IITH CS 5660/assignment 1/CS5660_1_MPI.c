#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#define ARR_SIZE 12

int main (int argc, char *argv[])
{
float param2;
int SIZE, rank;
int tag=1, param1;
int i=0;

MPI_Status StatusArr[ARR_SIZE];
MPI_Request RequestArr[ARR_SIZE];

MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &SIZE);

if (rank == 1) {
  for (i=0; i<ARR_SIZE; i++) {
    param1 = i*10;
    MPI_Isend(&param1, 1, MPI_INT, 1, tag, MPI_COMM_WORLD, &RequestArr[i]);
    MPI_Wait(&RequestArr[i], &StatusArr[i]);
    printf("Rank  %d sent value %d\n",rank,param1);
    }
  }

if (rank == 0) {
  for (i=0; i<ARR_SIZE; i++) {
    MPI_Irecv(&param2, 1, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &RequestArr[i]);
    MPI_Wait(&RequestArr[i], &StatusArr[i]);
    printf("Rank %d received value %f\n",rank,param2);
    }
  }

MPI_Finalize();
}
