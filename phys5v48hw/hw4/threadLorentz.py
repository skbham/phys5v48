# thread_lorentz.py
import threading

import argparse
import asyncio
import numpy as np
import os
import pandas as pd
from time import perf_counter
import tracemalloc

def add_chunk(n, counts, lock, bins=100, xmin=-10, xmax=10):
    """
    Generate n samples and add to global counts.
    """
    local_counts = lorentzian_histogram(n, bins, xmin, xmax)
    # Acquire lock to merge partial counts into global
    with lock:
        counts += local_counts

def run_threaded(n, n_threads=4, bins=100, xmin=-10, xmax=10):
    """
    Run the Lorentzian sampling in parallel using threads.
    """
    # Split n samples among processes
    chunks = (n // n_threads) * np.ones(n_threads, dtype=int)
    chunks[:n % n_threads] += 1 # Distribute remainder
    threads = [None] * n_threads # Thread list
    counts = np.zeros(bins) # Global counts
    lock = threading.Lock() # Lock for global data
    for i in range(n_threads):
        t = threading.Thread(target=add_chunk, args=(chunks[i], counts, lock, bins, xmin, xmax))
        t.start() # Start thread
        threads[i] = t
    for t in threads:
        t.join() # Wait for all threads to finish
    return counts

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

counts = run_threaded(args['n'], bins=args['bins'])

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

