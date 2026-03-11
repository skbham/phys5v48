#!/bin/bash
#SBATCH -J hw4Async
#SBATCH --output=hw4AsyncOut.txt
#SBATCH -p cpu-preempt
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

n=100000000
nodes=2
baseName="asyncLorentz"

for ((nP=1; nP<17; nP*=2)); do
for ((bins=10; bins<1001; bins*=10)); do

	fNameCounts = "${baseName}CountsNP${nP}B${bins}.txt"
	srun -n "$nP" python "${baseName}.py" "$n" "$bins" "$nP" "$nodes" "${baseName}Cat.xlsx" "$fNameCounts"

done
done

