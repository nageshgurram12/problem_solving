#include <omp.h>
#include <stdio.h>  
 main(int argc, char *argv[])  {

 int   i, n, chunk;
 int  a[100], b[100], result;

 /* Some initializations */
 n = 100;
 chunk = 10;
 result = 0.0;
 for (i=0; i < n; i++) {
   a[i] = i * 1;
   b[i] = i * 2;
   }

 #pragma omp parallel for      default(shared) private(i)    schedule(static,chunk)        reduction(+:result)
 for (i=0; i < n; i++)
     result = result + (a[i] * b[i]);

 printf("Final result= %d\n",result);

 }
