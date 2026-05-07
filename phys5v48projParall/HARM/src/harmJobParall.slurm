!/bin/bash
#SBATCH -J harmOpenMP
#SBATCH --output=harmOpenMPOutA.txt
#SBATCH -p cpu-preempt
#SBATCH -N 4 # nodes
#SBATCH -n 16 # total ranks
#SBATCH -c 1 # CPUs per rank
#SBATCH -t 02:00:00 # walltime
#SBATCH --exclusive # avoid interference

fnameTOut="harmOpenMPOut.txt"

start=$SECONDS

export OMP_NUM_THREADS=64

./harm "1"

duration=$(( SECONDS - start ))

echo "$duration" >> "$fnameTOut"
