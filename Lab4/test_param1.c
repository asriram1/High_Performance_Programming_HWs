/*************************************************************************/
// gcc -pthread test_param1.c -o test_param

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS 5

/************************************************************************/
void *PrintHello(void *threadid)
{

  //printf(" Hello  # %lu ! \n", threadid);
  // long unsigned int tid;
  // tid = (long unsigned int) threadid;
  signed char tid;
  tid = (signed char) threadid;

  printf(" Hello World!  It's me, thread # %s ! \n", tid);

  pthread_exit(NULL);
}

/*************************************************************************/
int main(int argc, char *argv[])
{
  int arg,i,j,k,m;   	              /* Local variables. */
  pthread_t threads[NUM_THREADS];
  int rc;
  long unsigned int t;
  signed char set;


  printf("\n Hello World!  It's me, MAIN!\n");

  for (t = 0; t < NUM_THREADS; t++) {
    printf("In main:  creating thread %ld\n", t);
    rc = pthread_create(&threads[t], NULL, PrintHello, (void*) set);
    if (rc) {
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }

  printf("\n It's me MAIN -- Good Bye World!\n");

  pthread_exit(NULL);

}/* end main */

