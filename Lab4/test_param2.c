/*************************************************************************/
// gcc -pthread test_param2.c -o test_param2

#include <stdio.h>
#include <stdlib.h>

#include <pthread.h>

#define NUM_THREADS 10
int global =0;

/********************/
void *work(void *i)
{
  long int j, k;
  //int f = *((int*)(i));  // get the value being pointed to
  int *g = (int*)(i);    // get the pointer itself


  for (j=0; j < 10000000; j++) k += j;  // busy work

  
  printf("\nHello World from %lu with value %d\n", pthread_self(), g[global]);
  g[global] *= 10;
  global += 1;

  //printf("\nHello World! %d  %d",  f, *g);

  pthread_exit(NULL);
}

/*************************************************************************/
int main(int argc, char *argv[])
{
  int arg,i,j,k,m;   	              /* Local variables. */
  pthread_t id[NUM_THREADS];
  int arr[NUM_THREADS] = {1,2,3,4,5,6,7,8,9,10}; 

  for (i = 0; i < NUM_THREADS; ++i) {
    if (pthread_create(&id[i], NULL, work, (void *)(&arr))) {
      printf("ERROR creating the thread\n");
      exit(19);
    }
  }

  for (j=0; j < 100000000; j++) k += j;  // busy work
  // int x = 0;
  // for(x = 0; x<NUM_THREADS; ++x){
  // printf("this is the array output: %d \n", arr[i]) ; 
  // }

  for( k = 0; k<NUM_THREADS; k++){

    printf("These are the final values of the array %d\n", arr[k]);
  }  
  
  printf("\nAfter creating the thread.  My id is %lu, i = %d\n",
	 (long)pthread_self(), i);

  return(0);

}/* end main */


