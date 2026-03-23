# thread_lorentz.py

import threading
import numpy as np
import pandas as pd
from time import perf_counter

# import custom modules
import invTransSamp # Import the set of functions

def add_chunk(n, counts, lock, bins=100, xmin=-10, xmax=10):
    """
    Generate n samples and add to global counts.
    """
    local_counts = invTransSamp.lorentzian_histogram(n, bins, xmin, xmax)
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

