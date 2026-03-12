#!/bin/bash
#SBATCH -J hw4Async
#SBATCH -A acc1
#SBATCH --output=hw4AsyncOut.txt
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
#bins = {10, 100, 1000}
#nP = {1,2,4,8,16,}
nodes = 2

baseName = "asyncLorentz"

for ((nP=1; nP<100; nP*=2))

    for ((bins=10, bins<=1000, bins*=10))

        fNameCounts = "${baseName}CountsNP${nP}B${bins}.txt"

        srun -n "$nP" python "${baseName}.py" --n "$n" --bins "$bins" --np "$nP" --nodes "$nodes" --fNameOut "${baseName}Cat.xlsx" --fNameCounts "$fNameCounts"
    
    end
end

