#include <unistd.h>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>

void function_1()
{
    for (int i = 0; i != 3; i++)
    {
        printf("Function 1 (i = %d)\n",i);
 	sleep(1); 	
    }
}

void function_2()
{
    for (int j = 0; j != 4; j++)
    {
	usleep(500 *1000);
	 printf("                   Function 2 (j = %d )\n", j);

    }
}
int main()
{
    #pragma omp parallel sections
    {    
        #pragma omp section
        function_1();
            
        #pragma omp section
        function_2();
    }

    return 0;
}
