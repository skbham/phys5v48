#!/bin/bash
#SBATCH -J hw5MPIShared
#SBATCH --output=hw5MPISharedOut.txt
#SBATCH -p cpu-preempt
#SBATCH -N 1 # nodes
#SBATCH -n 16 # total ranks
#SBATCH -c 1 # CPUs per rank
#SBATCH -t 00:01:00 # walltime
#SBATCH --exclusive # avoid interference

mpicxx -O3 -std=c++17 nbodyMPIShared.cc -o nbodyMPIShared

for ((N=128; N<=10000; N*=2)); do
    for ((tNum=1; tNum<=16; tNum*=2)); do

        time mpiexec -n "$tNum" ./nbodyMPIShared "$N" "$tNum"
        
    done
done

