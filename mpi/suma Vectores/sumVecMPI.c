#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define sizeVec 1000000                 /* size vec A,B,C */
#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

int main (int argc, char *argv[])
{
int	numtasks,              /* number of tasks in partition */
	taskid,                /* a task identifier */
	numworkers,            /* number of worker tasks */
	source,                /* task id of message source */
	dest,                  /* task id of message destination */
	mtype,                 /* message type */
	nelements,                  /* nelements of matrix A sent to each worker */
	averow, extra, offset, /* used to determine nelements sent to each worker */
	i, j, k, rc;           /* misc */
float	a[sizeVec],
	b[sizeVec],
	c[sizeVec];           /* result vector C */
MPI_Status status;

MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
if (numtasks < 2 ) {
  printf("Need at least two MPI tasks. Quitting...\n");
  MPI_Abort(MPI_COMM_WORLD, rc);
  exit(1);
  }
numworkers = numtasks-1;


/**************************** master task ************************************/
   if (taskid == MASTER)
   {
	    printf("mpi_mm has started with %d tasks.\n",numtasks);
	    printf("Initializing arrays...\n");
	    for (i=0; i<sizeVec; i++){
	        a[i]= i;
			b[i]= i+1;
		}


      /* Send vector data to the worker tasks */
      averow = sizeVec/numworkers;
      extra = sizeVec%numworkers;
      offset = 0;
      mtype = FROM_MASTER;
      for (dest=1; dest<=numworkers; dest++)
      {
         nelements = (dest <= extra) ? averow+1 : averow;
         printf("Sending %d nelements to task %d offset=%d\n",nelements,dest,offset);
         MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         MPI_Send(&nelements, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         MPI_Send(&a[offset], nelements, MPI_FLOAT, dest, mtype, MPI_COMM_WORLD);
         MPI_Send(&b[offset], nelements, MPI_FLOAT, dest, mtype, MPI_COMM_WORLD);
         offset = offset + nelements;
      }

      /* Receive results from worker tasks */
      mtype = FROM_WORKER;
      for (i=1; i<=numworkers; i++)
      {
         source = i;
         MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
         MPI_Recv(&nelements, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
         MPI_Recv(&c[offset], nelements, MPI_FLOAT, source, mtype, MPI_COMM_WORLD, &status);
         printf("Received results from task %d\n",source);
      }

      /* Print results */
      printf("******************************************************\n");
      printf("Result Vector:\n");
      for (i=0; i<sizeVec; i++)
      {
         printf(" %.2f ", c[i]);
      }
      printf("\n******************************************************\n");
      printf ("Done.\n");
   }


/**************************** worker task ************************************/
	if (taskid > MASTER)
   	{
	    mtype = FROM_MASTER;
	    MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
	    MPI_Recv(&nelements, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
	    MPI_Recv(&a, nelements, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD, &status);
	    MPI_Recv(&b, nelements, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD, &status);

	    for (k=0; k<nelements; k++)
	    	c[k]=a[k]+b[k];

	    mtype = FROM_WORKER;
	    MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
	    MPI_Send(&nelements, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
	    MPI_Send(&c, nelements, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD);
   	}
   	MPI_Finalize();
}
