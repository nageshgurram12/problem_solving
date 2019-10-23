#include <stdio.h>
#include <stdlib.h>

#include "mpi.h"

int main (int argc, char *argv[])
{
int SIZE, rank, dest, tag, source, rc, count;
char ReceivedMessage, MessageToSend='x';
MPI_Status Stat;

MPI_Init(&argc,&argv);
MPI_Comm_size(MPI_COMM_WORLD, &SIZE);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
printf("Rank %d starting...\n",rank);

if (rank == 0) {
  if (SIZE < 2)
  {printf("Need at least two ranks, Specify '-np 2' exiting \n");
      MPI_Abort(MPI_COMM_WORLD, rc);
   exit(0);
  }
  
  dest = rank + 1;
  source = dest;
  tag = rank;
  rc = MPI_Send(&MessageToSend, 1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
  printf("Sent to rank %d...\n",dest);
  rc = MPI_Recv(&ReceivedMessage, 1, MPI_CHAR, source, tag, MPI_COMM_WORLD, &Stat);
  printf("Received from rank %d...\n",source);
  }

else if (rank == 1) {
  dest = rank - 1;
  source = dest;
  tag = rank;
  rc = MPI_Recv(&ReceivedMessage, 1, MPI_CHAR, source, tag, MPI_COMM_WORLD, &Stat);
  printf("Received from rank %d...\n",source);
  rc = MPI_Send(&MessageToSend, 1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
  printf("Sent to rank %d...\n",dest);
  }

if (rank < 2) {
  rc = MPI_Get_count(&Stat, MPI_CHAR, &count);
  printf("Rank %d: Received %d char(s) from rank %d with tag %d \n",
         rank, count, Stat.MPI_SOURCE, Stat.MPI_TAG);
  }

MPI_Finalize();
}
