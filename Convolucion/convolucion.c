#include <stdio.h>
#include <cuda.h>
#include <time.h>

void inicializarVec(int *vec , int t){
  for(int i = 0; i < t ; i++)
    vec[i] = i + 1;
}

void convolucionCPU(int *h_n,int *h_mascara,int *h_r,int n, int mascara, int r){
	int mitadMascara= (mascara/2);
	for(int i=0;i<n;i++){
		int p=0;// almacena los valores temporales
    int k= i - mitadMascara;
		for (int j =0; j < mascara; j++){
      if(k < n  && k >= 0){
        p += h_n[k]*h_mascara[j];
      }
      else
        p+=0;
      k++;
		}
    h_r[i]=p;
	}
}

void imprimirVec(int *h_r,int n){
	for (int i = 0; i < n; i++)
		printf(" %d ",h_r[i]);
	printf("\n");
}

int main()
{
	int *h_n,*h_mascara,*h_r;

	// dimensiones
	int n= 10, mascara = 5, r=n;

	// asignacion de memoria
	h_n= (int*)malloc(n*sizeof(int));
	h_mascara= (int*)malloc(mascara*sizeof(int));
	h_r= (int*)malloc(r*sizeof(int));
	//inicializacion
	inicializarVec(h_n,n);
	inicializarVec(h_mascara,mascara);
	imprimirVec(h_n,n);
	imprimirVec(h_mascara,mascara);
	convolucionCPU(h_n,h_mascara,h_r,n,mascara,r);

	imprimirVec(h_r,r);
	return 0;
}
