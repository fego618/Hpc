#!/bin/bash

#SBATCH --job-name=mpiWithCuda_mm
#SBATCH --output=mpiWithCuda_mm.out
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

export CUDA_VISIBLE_DEVICES=0
mpicc -c mpi_mult_matriz.c -o mpi_mult_matriz.o
/usr/local/cuda/bin/nvcc cuda_mult_matriz.cu -c cuda_mult_matriz.o
mpicc mpi_mult_matriz.o cuda_mult_matriz.o -o mpiWithCuda_mm -L/usr/local/cuda/lib64/ -lcudart

