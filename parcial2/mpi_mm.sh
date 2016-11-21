#!/bin/bash

#SBATCH --job-name=mpi_mm
#SBATCH --output=mpi_mm.out
#SBATCH --nodes=4
#SBATCH --ntasks=8
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

mpicc -o mpi_multm mpi_mult_matriz.c
mpirun mpi_multm
