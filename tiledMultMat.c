#include <stdio.h>
#include <cuda.h>
#include <time.h>

#define N 1000
#define TILE_WIDTH 4
#define blocksize 4

__global__ void kernelMultMatTiled(float *d_M, float *d_N, float *d_P, int m,int n , int y){
  
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;

  float Pvalue = 0;

  for(int i = 0; i < n / TILE_WIDTH; i++){
   
    if((i*TILE_WIDTH + tx) < n && row < m){
      Mds[ty][tx]=d_M[row*n + (i*TILE_WIDTH + tx)];
    }else{
      Mds[ty][tx]=0.0;
    }

    if((i*TILE_WIDTH + ty) < n && col < y){
      Nds[ty][tx]= d_N[(i*TILE_WIDTH + ty)*y + col];
    }else{
      Nds[ty][tx]=0.0;
    }
    __syncthreads();

    for(int k = 0; k < TILE_WIDTH; ++k){
      Pvalue += Mds[ty][k] * Nds[k][tx];      
    }
    __syncthreads();
  }
  if(row<m && col < y)
  d_P[(row*y)+ col] = Pvalue;
}



__global__ void kernelMultMat(int *a, int *b, int *c,int m){
  int i,add; 
  int col=blockDim.x*blockIdx.x + threadIdx.x;
  int row=blockDim.y*blockIdx.y + threadIdx.y;
  if(col<m && row<m) {
    add=0;
    for(i=0; i< m ;i++){
        add += a[i+m*row]*b[col+m*i];  
   }
   c[row*m+col] = add;
  }
}

/*
  Nombre de la funcion: inicializarMat
  Parametros: Ma, m , n
    Ma: matriz que se va a inicializar
    m: numero de filas
    n: numero de columnas
  Objetivo: inicializar una matriz con valores "aleatorios"

*/
void inicializarMat(float *Ma , int m, int n){
  for(int i = 0; i <= m*n +1; i++){
    Ma[i] = 1;
  }
}

/*
  Nombre de la funcion: multiplicacionHost
  Parametros: h_Ma, h_Mb, h_Mc, m , n , y
    h_Ma : datos de la matriz A
    h_Mb : datos de la matriz B
    h_Mc : Matriz donde se van almacenar los datos
    m : Numero de filas de la matriz A
    n : Numero de columnas de la matriz A y numero de filas de la matriz B
    y : Numero de columnas de la matriz B

    Matriz h_Ma= m*n  Matriz h_Mb= n*y  Matriz h_Mc = m*y
  Objetivo: multiplicar dos matrices de diferentes dimensiones

*/
void multiplicacionHost(float *h_Ma, float *h_Mb, float *h_Mc, int m, int n, int y){
    float p;
    for(int row = 0; row < m ; row++){
      for(int col = 0; col < y ; col++){
        p = 0;
        for(int k = 0; k < n; k++){
          p += h_Ma[row*m+k] * h_Mb[k*n+col];
        }
        h_Mc[row*m+col] = p;
      }
    }
}

/*
  Nombre de la funcion: printData
  Parametros: Mat, m , n
    Mat: los valores de la matriz que se van a imprimir
    m: numero de filas
    n: numero de columnas
  Objetivo : imprimir los datos de una matriz

*/
int printData(float *Mat, int m,int n, int tipo){

  if(tipo == 1)
    printf("================ Matriz A ================ \n");
  if(tipo == 2)
    printf("================ Matriz B ================ \n");

  for(int i = 0; i < m; ++i){
    for(int j = 0; j < n; ++j){
      printf("%.2f ", Mat[(i*m)+j]);
    }
    printf("\n");
  }
  printf("=============================\n\n");
  return 0;
}


int main(){

  float *h_Ma,*h_Mb,*h_Mc,*d_Ma,*d_Mb,*d_Mc,*h_Mresult;
  clock_t start, end, startGPU, endGPU;
  double cpu_time_used, gpu_time_used;

  int n,m,y;
  //dimension de matrices m*n y n*y
  m=1600;
  n=1600;
  y=1500;

  //asignacion memoria en el host 
  
  h_Ma= (float*)malloc((m*n)*sizeof(float));
  h_Mb= (float*)malloc((n*y)*sizeof(float));
  h_Mc= (float*)malloc((m*y)*sizeof(float));
  h_Mresult = (float*)malloc((m*y)*sizeof(float));
  
  // inicializar matrices h_Ma, h_Mb
  inicializarMat(h_Ma,m,n);
  inicializarMat(h_Mb,n,y);
  //impresion de matrices iniciales
  //printData(h_Ma,m,n,1);
  //printData(h_Mb,n,y,2);
  //multiplicacion en CPU
  start = clock();
  multiplicacionHost(h_Ma,h_Mb,h_Mc,m,n,y);
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("Tiempo CPU: %.10f\n", cpu_time_used);


  cudaMalloc((void**)&d_Ma,(m*n)*sizeof(float));
  cudaMalloc((void**)&d_Mb,(n*y)*sizeof(float));
  cudaMalloc((void**)&d_Mc,(m*y)*sizeof(float));

  startGPU = clock();
  cudaMemcpy(d_Ma,h_Ma,(m*n)*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_Mb,h_Mb,(n*y)*sizeof(float),cudaMemcpyHostToDevice);
  
  dim3 DimBlock(blocksize, blocksize, 1);
  dim3 DimGrid(ceil(y / float(blocksize)), ceil(m / float(blocksize)), 1);

  kernelMultMatTiled<<<DimGrid,DimBlock>>>(d_Ma,d_Mb,d_Mc,m,n,y);
  cudaDeviceSynchronize();

  cudaMemcpy(h_Mresult,d_Mc,(m*y)*sizeof(float),cudaMemcpyDeviceToHost);
  endGPU = clock();
  gpu_time_used = ((double) (endGPU - startGPU)) / CLOCKS_PER_SEC;
  printf("Tiempo GPU : %.10f\n", gpu_time_used);
  printf("\n");
  /*
  printf("================ Matriz CPU ================\n");
  printData(h_Mc,m,y,0);
  printf("================ Matriz GPU ================\n");
  printData(h_Mresult,m,y, 0);
  */

  
  return 0;
}