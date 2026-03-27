#!/bin/bash
#SBATCH -J hw5OMP
#SBATCH --output=hw5OMPOut.txt
#SBATCH -p cpu-preempt
#SBATCH -N 1 # nodes
#SBATCH -n 16 # total ranks
#SBATCH -c 1 # CPUs per rank
#SBATCH -t 00:01:00 # walltime
#SBATCH --exclusive # avoid interference

g++ -O3 -std=c++17 -fopenmp nbodyOMP.cc -o nbodyOMP

for ((N=128; N<=10000; N*=2)); do
    for ((tNum=1; tNum<=16; tNum*=2)); do

        export OMP_NUM_THREADS=$tNum
        time ./nbodyOMP "$N" "$tNum"
        
    done
done

