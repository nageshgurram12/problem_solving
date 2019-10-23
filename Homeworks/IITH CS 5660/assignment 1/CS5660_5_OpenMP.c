#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[])
{
int nthreads, i, tid;
long int sum;

#pragma omp parallel
  {
  tid = omp_get_thread_num();

  if (tid == 0) {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  //printf("Thread %d is starting...\n",tid);

  #pragma omp barrier

//  sum = 0.0;
  #pragma omp for schedule(dynamic,10)
  for (i=0; i<1000; i++)
     sum = sum + i;

    printf ("Thread %d is done! Total= %ld\n",tid,sum);
    //This program may print different values of sum in every execution. Can you think why that bug is happening?
  }
}

 
