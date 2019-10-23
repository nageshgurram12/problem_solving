#include <mpi.h>
#include <stdio.h>

int main(int argc, char ** argv)
{
    int rank;
    MPI_Init(&argc, &argv);
    int count = 100;
    int sendbuf[100], recvbuf[100];
    int tag = 1;      
   MPI_Status status; 
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
   if (rank == 0) {
   MPI_Send (sendbuf, count, MPI_INT, 1, tag, MPI_COMM_WORLD);
   MPI_Recv (recvbuf, count, MPI_INT, 1, tag, MPI_COMM_WORLD, &status);
} else if (rank == 1) {
   MPI_Send (sendbuf, count, MPI_INT, 0, tag, MPI_COMM_WORLD);
   MPI_Recv (recvbuf, count, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
}
 MPI_Finalize();
    return 0;
}

