#include <cuda.h>
#include <cstdio>
#define BLOCKSIZE 32
__global__ void kernelMultMat(double *d_a, double *d_b, double *d_c, int ROWS, int COLA, int COLB) {
  
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  double add;

  if (row < ROWS && col < COLB) {
    add = 0;
    for (int k = 0; k < COLA; k++) {
      add += d_a[row * COLA + k] * d_b[k * COLB + col];
    }
    d_c[row * COLB + col] = add;
  }
}


void cuda_mult_matriz(double *h_a, double *h_b, double *h_c, int ROWS, int COLA, int COLB){
	
	double *d_a,*d_b,*d_c;

	int sizeA = ROWS*COLA;
	int sizeB = COLA*COLB;
	int sizeC = ROWS*COLB;

	cudaMalloc(&d_a, sizeof(double)*sizeA);
	cudaMalloc(&d_b, sizeof(double)*sizeB);
	cudaMalloc(&d_c, sizeof(double)*sizeC);

	cudaMemcpy(d_a,h_a,sizeA * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,h_b,sizeB * sizeof(double), cudaMemcpyHostToDevice);

	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
	dim3 dimGrid((COLB / BLOCKSIZE) + 1, (ROWS / BLOCKSIZE) + 1,1);

	kernelMultMat<<< dimGrid, dimBlock >>>(d_a, d_b, d_c, ROWS, COLA, COLB);
	cudaMemcpy(h_c, d_c, sizeC*sizeof(double),cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

}