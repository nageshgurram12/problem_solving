#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char ** argv)
{
    int rank, i, data[100], size, count; 
      MPI_Status status;      

     MPI_Init(&argc, &argv);
    int group_id = 0; 

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank != 0)  /* worker process */
        {
          srand(rank); // just to make sure every rank generates random number differently

          int numTasksDone = rand() % 100; 
          MPI_Send(data, numTasksDone, MPI_INT, 0, group_id,
                 MPI_COMM_WORLD);
          
          printf(" rank %d did %d tasks \n", rank, numTasksDone);
        }
    else {  /* master process */
          for (i = 0; i < size - 1; i++) {
            MPI_Recv(data, 100, MPI_INT, MPI_ANY_SOURCE,
                     MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_INT, &count);
            printf(" worker ID: %d; task ID: %d; count: %d\n",
                   status.MPI_SOURCE, status.MPI_TAG, count);
        }
    }

    MPI_Finalize();
}


