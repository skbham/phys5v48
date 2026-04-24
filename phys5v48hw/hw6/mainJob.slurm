#!/bin/bash
#SBATCH -J hw6
#SBATCH --output=hw6Out.txt
#SBATCH -p cpu-preempt
#SBATCH -N 1 # nodes
#SBATCH -n 16 # total ranks
#SBATCH -c 1 # CPUs per rank
#SBATCH -t 00:05:00 # walltime
#SBATCH --exclusive # avoid interference

python main.py

