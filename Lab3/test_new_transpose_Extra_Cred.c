/*****************************************************************************/
// gcc -O1 -mavx test_transpose.c -lrt -o test_transpose

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <xmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

#define GIG 1000000000
#define CPG 2.6           // Cycles per GHz -- Adjust to your computer

#define BASE  0
#define ITERS 31
#define DELTA 100
#define BSIZE 10
#define BBASE 16
#define BITERS 5

#define OPTIONS 1       // ij and ji -- need to add other methods

typedef double data_t;

/* Create abstract data type for vector */
typedef struct {
  long int len;
  data_t *data;
} vec_rec, *vec_ptr;

/*****************************************************************************/
int main(int argc, char *argv[])
{
  int OPTION;
  struct timespec diff(struct timespec start, struct timespec end);
  struct timespec time1, time2;
  struct timespec time_stamp[OPTIONS][ITERS+1];
  int clock_gettime(clockid_t clk_id, struct timespec *tp);
  vec_ptr new_vec(long int len);
  int set_vec_length(vec_ptr v, long int index);
  long int get_vec_length(vec_ptr v);
  int init_vector(vec_ptr v, long int len);
  data_t *data_holder;
  //void transpose(vec_ptr v0, vec_ptr v1);
  //void transpose_rev(vec_ptr v0, vec_ptr v1);
  //void transpose3(vec_ptr v0, vec_ptr v1);
  void transpose4(vec_ptr v0, vec_ptr v1, vec_ptr v2, vec_ptr v3, vec_ptr v4, vec_ptr v5, vec_ptr v6, vec_ptr v7);
  //inline void transpose4(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3, __m256 &row4);

  long int i, j, k, bsize = BSIZE;
  long int time_sec, time_ns;
  long int MAXSIZE = BASE+(ITERS+1)*DELTA;

  printf("\n Hello World -- Transpose \n");

  // declare and initialize the vector structures
  vec_ptr v0 = new_vec(MAXSIZE);  vec_ptr v1 = new_vec(MAXSIZE);
  vec_ptr v2 = new_vec(MAXSIZE); vec_ptr v3 = new_vec(MAXSIZE);
  vec_ptr v4 = new_vec(MAXSIZE);  vec_ptr v5 = new_vec(MAXSIZE); vec_ptr v6 = new_vec(MAXSIZE); vec_ptr v7 = new_vec(MAXSIZE);
  init_vector(v0, MAXSIZE);  init_vector(v1, MAXSIZE); init_vector(v2, MAXSIZE);  init_vector(v3, MAXSIZE);
  init_vector(v4, MAXSIZE);  init_vector(v5, MAXSIZE); init_vector(v6, MAXSIZE);  init_vector(v7, MAXSIZE);
/*
  OPTION = 0;
  for (i = 0; i < ITERS; i++) {
    set_vec_length(v0,BASE+(i+1)*DELTA);
    set_vec_length(v1,BASE+(i+1)*DELTA);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
    transpose(v0, v1);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
    time_stamp[OPTION][i] = diff(time1,time2);
  }

  OPTION++;
  for (i = 0; i < ITERS; i++) {
    set_vec_length(v0,BASE+(i+1)*DELTA);
    set_vec_length(v1,BASE+(i+1)*DELTA);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
    transpose_rev(v0, v1);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
    time_stamp[OPTION][i] = diff(time1,time2);
  }

  OPTION++;
  for (i = 0; i < ITERS; i++) {
    set_vec_length(v0,BASE+(i+1)*DELTA);
    set_vec_length(v1,BASE+(i+1)*DELTA);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
    transpose3(v0, v1);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
    time_stamp[OPTION][i] = diff(time1,time2);
  }
  */
  OPTION = 0;
  for (i = 0; i < ITERS; i++) {
    set_vec_length(v0,BASE+(i+1)*DELTA);
    set_vec_length(v1,BASE+(i+1)*DELTA);
    set_vec_length(v2,BASE+(i+1)*DELTA);
    set_vec_length(v3,BASE+(i+1)*DELTA);
    set_vec_length(v4,BASE+(i+1)*DELTA);
    set_vec_length(v5,BASE+(i+1)*DELTA);
    set_vec_length(v6,BASE+(i+1)*DELTA);
    set_vec_length(v7,BASE+(i+1)*DELTA);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
    transpose4(v0, v1, v2, v3, v4, v5, v6, v7);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
    time_stamp[OPTION][i] = diff(time1,time2);
  }
  /* output times */
  printf("\nsize,   ij,   ji, block = %d", BSIZE);  
  for (i = 0; i < ITERS; i++) {
    printf("\n%ld, ", BASE+(i+1)*DELTA);
    for (j = 0; j < OPTIONS; j++) {
      if (j != 0) printf(", ");
      printf("%ld", (long int)((double)(CPG)*(double)
     (GIG * time_stamp[j][i].tv_sec + time_stamp[j][i].tv_nsec)));
    }
  }

  printf("\n");
  
}/* end main */
/*********************************/

/* Create 2D vector of specified length per dimension */
vec_ptr new_vec(long int len)
{
  long int i;

  /* Allocate and declare header structure */
  vec_ptr result = (vec_ptr) malloc(sizeof(vec_rec));
  if (!result) return NULL;  /* Couldn't allocate storage */
  result->len = len;

  /* Allocate and declare array */
  if (len > 0) {
    data_t *data = (data_t *) calloc(len*len, sizeof(data_t));
    if (!data) {
    free((void *) result);
    printf("\n COULDN'T ALLOCATE %ld BYTES STORAGE \n", result->len);
    return NULL;  /* Couldn't allocate storage */
  }
  result->data = data;
  }
  else result->data = NULL;

  return result;
}

/* Set length of vector */
int set_vec_length(vec_ptr v, long int index)
{
  v->len = index;
  return 1;
}

/* Return length of vector */
long int get_vec_length(vec_ptr v)
{
  return v->len;
}

/* initialize 2D vector */
int init_vector(vec_ptr v, long int len)
{
  long int i;

  if (len > 0) {
    v->len = len;
    for (i = 0; i < len*len; i++) v->data[i] = (data_t)(i);
    return 1;
  }
  else return 0;
}

data_t *get_vec_start(vec_ptr v)
{
  return v->data;
}

/************************************/

struct timespec diff(struct timespec start, struct timespec end)
{
  struct timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return temp;
}


void transpose4(vec_ptr v0, vec_ptr v1, vec_ptr v2, vec_ptr v3, vec_ptr v4, vec_ptr v5, vec_ptr v6, vec_ptr v7)
{
  long int i, j, k;
  __m256d   m0, m1, m2, m3, m4, m5, m6, m7;

  data_t *data[8];
  
    data[0] = get_vec_start(v0);
    data[1] = get_vec_start(v1);
    data[2] = get_vec_start(v2);
    data[3] = get_vec_start(v3);
    data[4] = get_vec_start(v4);
    data[5] = get_vec_start(v5);
    data[6] = get_vec_start(v6);
    data[7] = get_vec_start(v7);

    __m256d*  pSrc[8];

  for(k = 0; k<8; k++){
    pSrc[i] = (__m256d*)(data[i]);
  }

  long int get_vec_length(vec_ptr v);
  long int length = get_vec_length(v0)/4;

  for (i = 0; i < 4; i++){

    for(j = 0; j<length; j++){
         m0 = (*pSrc[i]);
         pSrc[i]++;
        m1 = (*pSrc[i]);
         pSrc[i]++;
        m2 = (*pSrc[i]);
         pSrc[i]++;
         m3 = (*pSrc[i]);
         pSrc[i]++;

         __m256d tmp3, tmp2, tmp1, tmp0; 
                    tmp0 = _mm256_shuffle_pd((m0),(m1), 0x0);                    
                    tmp2 = _mm256_shuffle_pd((m0),(m1), 0xF);                
                    tmp1 = _mm256_shuffle_pd((m2),(m3), 0x0);                    
                    tmp3 = _mm256_shuffle_pd((m2),(m3), 0xF);                
                                                                                 
                    (m0) = _mm256_permute2f128_pd(tmp0, tmp1, 0x20);   
                    (m1) = _mm256_permute2f128_pd(tmp2, tmp3, 0x20);   
                    (m2) = _mm256_permute2f128_pd(tmp0, tmp1, 0x31);   
                    (m3) = _mm256_permute2f128_pd(tmp2, tmp3, 0x31);     


          (*pSrc[i+4]) = m0;
          pSrc[i+4]++;
          (*pSrc[i+4]) = m1;
          pSrc[i+4]++;
          (*pSrc[i+4]) = m2;
          pSrc[i+4]++;
          (*pSrc[i+4]) = m3;
          pSrc[i+4]++;

      }

  }

}