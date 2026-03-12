# mpi_lorentz.py
from mpi4py import MPI

import argparse
import asyncio
import numpy as np
import os
import pandas as pd
from time import perf_counter
import tracemalloc

def lorentzian_histogram(n, bins=100, xmin=-10, xmax=10, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    u = rng.random(n)
    x = 1. / np.tan(np.pi * u)
    counts, _ = np.histogram(x, bins=bins, range=(xmin, xmax))
    return counts.astype(np.int64)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Start timing
if rank == 0:
    import time
    start_time = time.time()

# Independent RNG stream per rank
seed = 42
ss = np.random.SeedSequence(seed)
child = ss.spawn(size)[rank]
rng = np.random.default_rng(child)

n_total = 10_000_000
chunks = np.full(size, n_total // size, dtype=int)
chunks[: n_total % size] += 1

local = lorentzian_histogram(int(chunks[rank]), bins=100, xmin=-10, xmax=10, rng=rng)
global_counts = np.empty_like(local)
comm.Allreduce(local, global_counts, op=MPI.SUM)

if rank == 0:
    import time
    end_time = time.time()
    print(f"Total samples: {n_total}")
    print(f"Runtime: {end_time - start_time:.3f} seconds")
    print(f"Samples per second: {n_total / (end_time - start_time):.0f}")

    # Save results
    bin_edges = np.linspace(-10, 10, 101)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    np.savetxt("lorentzian_histogram.txt",
                np.column_stack([bin_centers, global_counts]),
                fmt="%.6f %d")
    print("Results saved to lorentzian_histogram.txt")

# Initialize the parser
parser = argparse.ArgumentParser()

# Add arguments to the parser
parser.add_argument('n', type=int)
parser.add_argument('bins', type=int)
parser.add_argument('nP', type=int)
parser.add_argument('nodes', type=int)
parser.add_argument('fNameOut', type=str)
parser.add_argument('fNameCounts', type=str)

# Get the input arguments
args = vars(parser.parse_args())	

tracemalloc.start() # Start monitoring memory
start = perf_counter() # Start timer

counts = run_aync(args['n'], bins=args['bins'])

end = perf_counter() # Stop timer
tracemalloc.stop() # Stop monitoring memory

t = end - start # Calculate time

if not os.path.exists(args['fNameOut']):
    file = open(args['fNameOut'], 'w')
    file.close()

df = pd.read_excel(args['fNameOut']) # Read in catalog

# Add entry to catalog
df.loc[-1] = [args['n'], args['bins'], args['nodes'], args['nP'], args['nP']/args['nodes'], t, tracemalloc.get_traced_memory()[1]]

# Write to the catalog
df.to_excel(args['fNameOut'], columns=["Problem Size (n)", "Bins", "Nodes", "Ranks", "Threads", "Threads Per Rank", "Runtime", "Peak Memory"])

# Save the counts to a .txt file
np.savetxt(args['fNameCounts'], np.array(counts))
