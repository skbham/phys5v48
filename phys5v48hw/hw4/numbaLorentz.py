# numba_lorentz.py

from numba import njit, prange, cuda
import numpy as np

@njit(parallel=True, nogil=True, nopython=False)

def lorentzian_histogram_numba(n, bins=100, xmin=-10, xmax=10):

    """
    Sample n random points from the Lorentzian distribution
    using inverse transform sampling. Make a histogram with
    the specified bin count and range. Returns counts.
    """
    
    xfac = bins / (xmax - xmin) # Factor to map x to bin index
    counts = np.zeros(bins) # Initialize counts
    for i in prange(n):
        u = np.random.random() # Uniform(0,1)
        x = 1. / np.tan(np.pi * u) # x = 1/tan(pi*u)
        ix = int((x - xmin) * xfac) # Map x to bin index
        if 0 <= ix < bins:
            cuda.atomic.add(counts, ix, 1) # Atomic increment

    return counts

