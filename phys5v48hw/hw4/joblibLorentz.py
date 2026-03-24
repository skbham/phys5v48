# joblib_lorentz.py

from joblib import Parallel, delayed
from time import perf_counter
import numpy as np
import pandas as pd
from time import perf_counter

# import custom modules
import invTransSamp # Import the set of functions

def run_joblib(n, n_jobs=4, bins=100, xmin=-10, xmax=10):
    """
    Run the Lorentzian sampling in parallel using joblib.
    """
    # Split n samples among jobs
    chunks = (n // n_jobs) * np.ones(n_jobs, dtype=int)
    chunks[:n % n_jobs] += 1 # Distribute remainder
    results = Parallel(n_jobs=n_jobs)(
        delayed(invTransSamp.lorentzian_histogram)(chunk, bins, xmin, xmax)
        for chunk in chunks
    )
    return np.sum(results, axis=0) # Aggregate results

