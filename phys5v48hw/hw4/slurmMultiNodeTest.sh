#!/bin/bash
# SBATCH -J hw4Async
# SBATCH --output=hw4AsyncOut.txt
# SBATCH -p cpu-preempt
# SBATCH -N 2 # nodes
# SBATCH -n 16 # total ranks
# SBATCH -c 1 # CPUs per rank
# SBATCH -t 00:10:00 # walltime
# SBATCH --exclusive # avoid interference
module purge
module load python mpi

# Avoid oversubscription by BLAS/NumPy threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

source /home/biscuit/Documents/utd/phys5v48/venv/bin/activate

n=1000
nodes=2
# baseName="asyncLorentz"
baseName="joblibLorentz" # Threading
fNameCat="output/${baseName}/${baseName}Cat.xlsx"

for ((nP=1; nP<17; nP*=2)); do
for ((bins=10; bins<1001; bins*=10)); do

	fNameCounts="output/${baseName}/${baseName}CountsNP${nP}B${bins}.txt"

	python "main.py" "$n" "$bins" "$nP" "$nodes" "$fNameCat" "$fNameCounts"

done
done


