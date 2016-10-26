/******************************************************************************
* FILE: omp_workshare1.c
* DESCRIPTION:
*   OpenMP Example - Loop Work-sharing - C/C++ Version
*   In this example, the iterations of a loop are scheduled dynamically
*   across the team of threads.  A thread will perform CHUNK iterations
*   at a time before being scheduled for the next CHUNK of work.
* AUTHOR: Blaise Barney  5/99
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define CHUNKSIZE   5
#define N       20

int main (int argc, char *argv[])
{
int nthreads, tid, i, chunk;
int a[N*N], b[N*N], c[N*N];

/* Some initializations */
for (i=0; i < N*N; i++){
  a[i] = 1;
  b[i] = 2;
}

chunk = CHUNKSIZE;

#pragma omp parallel shared(a,b,c,nthreads,chunk) private(i,tid)
  {
  tid = omp_get_thread_num();
  if (tid == 0)
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d starting...\n",tid);

  #pragma omp for schedule(dynamic,chunk)
  for (i=0; i<N; i++){
    for(int j=0;j<N;j++)
    {
      c[i*N + j] = a[i*N+j] + b[i*N+j];
    printf("Thread %d: c[%d]= %d\n",tid,i,c[i]);
    }
  }

  }  /* end of parallel section */

}
