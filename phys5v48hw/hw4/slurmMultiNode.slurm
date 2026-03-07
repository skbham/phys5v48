#!/bin/bash
#SBATCH -J hw4-lorentz
#SBATCH -A your_account
#SBATCH -p compute
#SBATCH -N 2 # nodes
#SBATCH -n 16 # total ranks
#SBATCH -c 1 # CPUs per rank
#SBATCH -t 00:10:00 # walltime
#SBATCH --exclusive # avoid interference
module purge
module load python mpi

# Avoid oversubscription by BLAS/NumPy threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

n = 100000000
bins = {10, 100, 1000}
nP = {1,2,4,8,16,}

for ((nP=1; nP<100; nP*=2))

    for ((bins=10, bins<=1000, bins*=10))

        srun -n "$nP" python mpi_lorentz.py --n 10000000 --bins "$bins"
    
    end
end

