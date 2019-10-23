#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define PI 3.1415926535
#define DELTA .01415926535
#define SIZE 1000000

int main (int argc, char *argv[])
{
int nthreads, tid, i;
float a[SIZE], b[SIZE];
omp_lock_t lock1, lock2;


omp_init_lock(&lock1);
omp_init_lock(&lock2);

/* Initialize the arrays */
for (i=0; i<SIZE; i++) {
  a[i]=0;
  b[i]=0;
  }

/* Fork a team of threads giving them their own copies of variables */
#pragma omp parallel shared(a, b, nthreads, lock1, lock2) private(tid, i)
  {

  /* Obtain thread number and number of threads */
  tid = omp_get_thread_num();
  #pragma omp master
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d starting...\n", tid);
  #pragma omp barrier

  #pragma omp sections nowait
    {
    #pragma omp section
      {
      omp_set_lock(&lock1);
      printf("Thread %d updating a[]\n",tid);
      for (i=0; i<SIZE; i++)
        a[i] += DELTA * i;
      omp_set_lock(&lock2);
      printf("Thread %d updating b[]\n",tid);
      for (i=0; i<SIZE; i++)
        b[i] += DELTA + i;
      omp_unset_lock(&lock2);
      omp_unset_lock(&lock1);
     }

    #pragma omp section
      {
      omp_set_lock(&lock1);

      for (i=0; i<SIZE; i++)
        b[i] += PI * i;
      omp_set_lock(&lock2);

      for (i=0; i<SIZE; i++)
        a[i] += PI + i;
      omp_unset_lock(&lock2);
      omp_unset_lock(&lock1);
      }

    }  /* end of sections */
  }  /* end of parallel region */
}
                  