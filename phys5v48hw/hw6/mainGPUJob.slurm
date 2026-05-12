#!/bin/bash
#SBATCH -J hw6GPU
#SBATCH --output=hw6GPUOut.txt
#SBATCH -p gpu-preempt
#SBATCH -N 1 # nodes
#SBATCH -n 16 # total ranks
#SBATCH -c 1 # CPUs per rank
#SBATCH -t 00:05:00 # walltime
#SBATCH --exclusive # avoid interference

python mainGPU.py

