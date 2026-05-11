#!/bin/bash
#SBATCH -J hw4Numba
#SBATCH --output=hw4NumbaOut.txt
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

# baseName="asyncLorentz" # AsyncIO
# baseName="threadLorentz" # Threading
# baseName="mpLorentz" # Multiprocessing
# baseName="ppeLorentz" # ProcessPoolExecutor
# baseName="daskLorentz" # Dask
# baseName="joblibLorentz" # JobLib
# baseName="mpireLorentz" # Mpire
baseName="numbaLorentz" # Numba

fNameCat="output/${baseName}/${baseName}Cat.xlsx"

for ((nP=1; nP<10000; nP*=2)); do
for ((bins=10; bins<1001; bins*=10)); do

	fNameCounts="output/${baseName}/${baseName}CountsNP${nP}B${bins}.txt"

	python "main.py" "$n" "$bins" "$nP" "$nodes" "$fNameCat" "$fNameCounts"

done
done

