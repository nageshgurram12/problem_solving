#include <stdio.h>  
#include <omp.h>  
int main()
{
    const int SIZE = 10;
    int a[10] = { 5, 4, 6, 10, 100, 34, 4300, 11, 22, 33};
    int max, i; 
    
max = a[0];   //a is an array
    #pragma omp parallel for num_threads(4)  
        for (i = 1; i < SIZE; i++) {  
            if (a[i] > max) {  
                #pragma omp critical  
                {  
                    // compare a[i] and max again because max   
                    // could have been changed by another thread after   
                    // the comparison outside the critical section  
                    if (a[i] > max)  
                        max = a[i];  
                }  
            }  
        }    
    printf("max = %d\n", max); 

}
