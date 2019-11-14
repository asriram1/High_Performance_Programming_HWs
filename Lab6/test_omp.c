/*********************************************************************/
// gcc -O1 -fopenmp test_omp.c -lrt -lm -o test_omp
// export NUM_OMP_THREADS=4
//

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define THREADS 4

/*********************************************************************/
int main(int argc, char *argv[])
{
  printf("\n Hello World -- Test OMP \n");

  omp_set_num_threads(THREADS);
  printf("Using %d threads for OpenMP\n", THREADS);

  printf("Printing 'Hello world!' using omp parallel and omp sections\n");

#pragma omp parallel
#pragma omp sections
  {
    //    printf("\n");
    //#pragma omp section
    printf("H");
#pragma omp section
    printf("e");
#pragma omp section
    printf("l");
#pragma omp section
    printf("l");
#pragma omp section
    printf("o");
#pragma omp section
    printf(" ");
#pragma omp section
    printf("W");
#pragma omp section
    printf("o");
#pragma omp section
    printf("r");
#pragma omp section
    printf("l");
#pragma omp section
    printf("d");
#pragma omp section
    printf("!");
  }

  printf("\n\n");

  printf("Printing 'Hello world!' using omp parallel for\n");

  /* ============ ADD YOUR CODE HERE ============ */

int N = 12;
int i;
char string[12] = "Hello World!";
//{"H","e","l","l", "o"," ","W", "o","r","l","d","!"};

#pragma omp parallel for

for(i = 0; i<N;i++){
  printf("%c", string[i]);
} 

printf("\n");


}/* end main */

/**********************************************/
