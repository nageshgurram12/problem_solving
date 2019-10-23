#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

int main(int argc, char *argv[]) {
  if (omp_in_parallel())
   {
    printf ("We are in parallel region \n");
   }
  else
    printf ("We are not in parallel region \n");

   printf("\n");
    printf ("omp_get_max_threads() %d \n", omp_get_max_threads());

   printf("==================================\n");
#pragma omp parallel
  {
   if (omp_in_parallel())
   {
    printf ("We are in parallel region \n");
   }
  else
    printf ("We are not in parallel region \n");
  
    omp_set_num_threads(2);
    
    printf ("omp_get_max_threads() %d \n", omp_get_max_threads());
    printf(" omp_get_num_threads() %d ", omp_get_num_threads());

    printf("Hello World from thread %d of %d\n", omp_get_thread_num(), omp_get_num_threads());          
  }

   return(0);
}

