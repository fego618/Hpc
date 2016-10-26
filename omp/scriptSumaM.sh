#!/bin/bash
#SBATCH --job-name sumaMEstiven
#SBATCH --nodes 1
#SBATCH --output MatrizOut.out
export OMP_NUM_THREADS=8
./sumaMatriz
