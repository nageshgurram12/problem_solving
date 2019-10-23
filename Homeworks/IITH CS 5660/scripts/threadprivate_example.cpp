#include <omp.h> 
 #include <stdio.h>  
 int  a, b, i, tid;
 float x;
 
 #pragma omp threadprivate(a, x)
 
 main(int argc, char *argv[]) {
 
   /* Explicitly turn off dynamic threads */
   omp_set_dynamic(0);
  x = 5;
   printf("1st Parallel Region:\n");
 //#pragma omp parallel private(b,tid) copyin(x)
   #pragma omp parallel private(b,tid) 
   {
   tid = omp_get_thread_num();
   
   printf("Thread %d:   a,b,x= %d %d       %f\n",tid,a,b,x);    
   
   a = tid;
   b = tid;
   x = 1.1 * tid +1.0;
   printf("Thread %d:   a,b,x= %d %d       %f\n\n",tid,a,b,x);
   }  /* end of parallel region */
 
   printf("************************************\n");
   printf("Master thread doing serial work here\n");
   printf("************************************\n");
 
   printf("2nd Parallel Region:\n");
 #pragma omp parallel private(tid) 
   {
   tid = omp_get_thread_num();
   printf("Thread %d:   a,b,x= %d %d %f\n",tid,a,b,x);
   }  /* end of parallel region */

 }


