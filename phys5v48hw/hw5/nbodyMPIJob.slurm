#!/bin/bash
#SBATCH -J hw5MPI
#SBATCH --output=hw5MPIOut.txt
#SBATCH -p cpu-preempt
#SBATCH -N 1 # nodes
#SBATCH -n 16 # total ranks
#SBATCH -c 1 # CPUs per rank
#SBATCH -t 00:01:00 # walltime
#SBATCH --exclusive # avoid interference

mpicxx -O3 -std=c++17 nbodyMPI.cc -o nbodyMPI

for ((N=128; N<=10000; N*=2)); do
    for ((tNum=1; tNum<=16; tNum*=2)); do

        time mpiexec -n "$tNum" ./nbodyMPI "$N" "$tNum"
        
    done
done

