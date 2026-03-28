#!/bin/bash
#SBATCH -J hw5MPIOMP
#SBATCH --output=hw5MPIOMPOut.txt
#SBATCH -p cpu-preempt
#SBATCH -N 2 # nodes
#SBATCH -n 16 # total ranks
#SBATCH -c 1 # CPUs per rank
#SBATCH -t 00:15:00 # walltime
#SBATCH --exclusive # avoid interference

mpicxx -fopenmp -O3 -std=c++17 nbodyMPIOMP.cc -o nbodyMPIOMP

export OMP_NUM_THREADS=16
time mpiexec -n 2 ./nbodyMPI "$N" 2

