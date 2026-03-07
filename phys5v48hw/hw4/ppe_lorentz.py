# ppe_lorentz.py
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

def run_ppe(n, max_workers=4, bins=100, xmin=-10, xmax=10):
    """
    Run the Lorentzian sampling in parallel using ProcessPoolExecutor.
    """
    chunks = (n // max_workers) * np.ones(max_workers, dtype=int) # Split n samples among workers
    chunks[:n % max_workers] += 1 # Distribute remainder
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(lorentzian_histogram, chunk, bins, xmin, xmax) for chunk in chunks]
        results = [f.result() for f in futures] # Collect results
    return np.sum(results, axis=0) # Aggregate results
