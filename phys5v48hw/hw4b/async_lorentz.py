# async_lorentz.py
import asyncio
import pandas as pd
from time import perf_counter
import tracemalloc

async def async_lorentzian_histogram(n, bins=100, xmin=-10, xmax=10):
    """
    Async wrapper for lorentzian_histogram. Since lorentzian_histogram
    is CPU-bound and synchronous, we call it directly.
    """
    return lorentzian_histogram(n, bins, xmin, xmax)

def lorentzian_histogram(n, bins=100, xmin=-10, xmax=10):
    """
    Sample n random points from the Lorentzian distribution
    using inverse transform sampling. Make a histogram with
    the specified bin count and range. Returns counts.
    """
    u = np.random.random(n) # Uniform(0,1)
    x = 1. / np.tan(np.pi * u) # x = 1/tan(pi*u)
    counts, _ = np.histogram(x, bins=bins, range=(xmin, xmax))
    return counts # No need to return bin edges for uniform bins

async def add_chunk(n, counts, bins=100, xmin=-10, xmax=10, n_subchunks=10):
    """
    Generate n samples in subchunks and add to global counts.
    """
    # Split n samples among sub-chunks
    sub_chunks = (n // n_subchunks) * np.ones(n_subchunks, dtype=int)
    sub_chunks[:n % n_subchunks] += 1 # Distribute remainder
    # Gather results from subchunks
    local_counts = await asyncio.gather(*[
        async_lorentzian_histogram(chunk, bins, xmin, xmax)
        for chunk in sub_chunks
    ])
    counts += np.sum(local_counts, axis=0) # Merge partial counts

async def get_counts(n, n_tasks=4, bins=100, xmin=-10, xmax=10, n_subchunks=10):
    """
    Async function to run the Lorentzian sampling in parallel using asyncio.
    """
    # Split n samples among tasks
    chunks = (n // n_tasks) * np.ones(n_tasks, dtype=int)
    chunks[:n % n_tasks] += 1 # Distribute remainder
    counts = np.zeros(bins) # Global counts
    tasks = [
        asyncio.create_task(add_chunk(chunk, counts, bins, xmin, xmax, n_subchunks))
        for chunk in chunks
    ]
    await asyncio.gather(*tasks) # Wait for all tasks to finish
    
    return counts

def run_async(n, n_tasks=4, bins=100, xmin=-10, xmax=10, n_subchunks=10):
    """
    Run the Lorentzian sampling in parallel using asyncio.
    """
    return asyncio.run(get_counts(n, n_tasks, bins, xmin, xmax, n_subchunks))
