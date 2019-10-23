#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

//Objective of this program: to illustrate that "private(b)"  does not initialize the value of variable "b".
const int size =10;

int main(int argc, char *argv[]) {
    int a=5, b=4; 

#pragma omp parallel private (b) default (shared)
  {     
     int tid = omp_get_thread_num();      
      printf("Value of b in thread %d is %d\n", tid, b);      
      printf("a is %d\n",a);
  }

   return(0);
}

