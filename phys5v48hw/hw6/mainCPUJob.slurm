#!/bin/bash
#SBATCH -J hw6CPU
#SBATCH --output=hw6CPUOut.txt
#SBATCH -p cpu-preempt
#SBATCH -N 1 # nodes
#SBATCH -n 1 # total ranks
#SBATCH -c 1 # CPUs per rank
#SBATCH -t 00:05:00 # walltime
#SBATCH --exclusive # avoid interference

python mainCPU.py

