#!/bin/bash
#SBATCH -J hw5Serial
#SBATCH --output=hw5SerialOut.txt
#SBATCH -p cpu-preempt
#SBATCH -N 1 # nodes
#SBATCH -n 16 # total ranks
#SBATCH -c 1 # CPUs per rank
#SBATCH -t 00:10:00 # walltime
#SBATCH --exclusive # avoid interference

g++ -O3 -std=c++17 nbodySerial.cc -o nbodySerial

for ((N=128; N<5000; N*=2)); do

    time ./nbodySerial "$N"

	#fNameCounts="output/${baseName}/${baseName}CountsNP${nP}B${bins}.txt"
	#python "main.py" "$n" "$bins" "$nP" "$nodes" "$fNameCat" "$fNameCounts"

done
