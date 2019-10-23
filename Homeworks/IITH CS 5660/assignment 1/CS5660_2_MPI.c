#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#define  Root		0

int main (int argc, char *argv[])
{
int  SIZE, rank, len, partner, message;
char hostname[MPI_MAX_PROCESSOR_NAME];
MPI_Status status;
 
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &SIZE);

/* need an even number of tasks  */
if (SIZE % 2 != 0) {
   if (rank == Root) 
      printf("I need an even number of ranks: SIZE=%d\n", SIZE);
   } 

else {
   if (rank == Root) 
      printf("Root: Number of MPI ranks is: %d\n",SIZE);

   MPI_Get_processor_name(hostname, &len);
   printf ("Hello from task %d on %s!\n", rank, hostname);

   /* determine partner and then send/receive with partner */
   if (rank < SIZE/2) {
     partner = SIZE/2 + rank;
     MPI_Send(&rank, 1, MPI_INT, partner, 1, MPI_COMM_WORLD);
     MPI_Recv(&message, 1, MPI_INT, partner, 1, MPI_COMM_WORLD, &status);
     }
   else if (rank >= SIZE/2) {
     partner = rank - SIZE/2;
     MPI_Recv(&message, 1, MPI_INT, partner, 1, MPI_COMM_WORLD, &status);
     MPI_Send(&rank, 1, MPI_INT, partner, 1, MPI_COMM_WORLD);
     }
    
   printf("Rank %d is partner with %d\n",rank,message);
   }
 
}
