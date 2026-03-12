# dask_lorentz.py
import dask
from dask import delayed

@delayed
def delayed_lorentzian_histogram(n, bins=100, xmin=-10, xmax=10):
    """
    Delayed function for lorentzian_histogram.
    """
    return lorentzian_histogram(n, bins, xmin, xmax)

def run_dask(n, n_tasks=4):
    """
    Run the Lorentzian sampling in parallel using Dask.
    """
    # Split n samples among tasks
    chunks = (n // n_tasks) * np.ones(n_tasks, dtype=int)
    chunks[:n % n_tasks] += 1 # Distribute remainder
    tasks = [delayed_lorentzian_histogram(chunk) for chunk in chunks]
    results = dask.compute(*tasks) # Compute all tasks
    return np.sum(results, axis=0) # Aggregate results
