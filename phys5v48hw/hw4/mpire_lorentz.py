# mpire_lorentz.py
from mpire import WorkerPool
from time import perf_counter

def run_mpire(n, n_jobs=4, bins=100, xmin=-10, xmax=10):
    """
    Run the Lorentzian sampling in parallel using mpire.
    """
    # Split n samples among jobs
    chunks = (n // n_jobs) * np.ones(n_jobs, dtype=int)
    chunks[:n % n_jobs] += 1 # Distribute remainder
    with WorkerPool(n_jobs=n_jobs) as pool:
        # See mpire docs for argument passing; alternatively use starmap
        results = pool.map(lorentzian_histogram, chunks)
    return np.sum(results, axis=0) # Aggregate results
