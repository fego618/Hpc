#!/bin/bash

#SBATCH --job-name=add_vector
#SBATCH --output=mpiSumVec.out
#SBATCH --nodes=6
#SBATCH --ntasks=8
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

mpicc -o sumVec sumVecMPI.c
mpirun sumVec
