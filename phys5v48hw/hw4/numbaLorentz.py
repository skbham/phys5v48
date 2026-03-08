# numba_lorentz.py
import numpy as np
from numba import njit, prange, atomic
from time import perf_counter

@njit(parallel=True, nogil=True)
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
            atomic.add(counts, ix, 1) # Atomic increment
    return counts

# Initialize the parser
parser = argparse.ArgumentParser()

# Add arguments to the parser
parser.add_argument('n', type=int)
parser.add_argument('bins', type=int)
parser.add_argument('nP', type=int)
parser.add_argument('nodes', type=int)
parset.add_argument('fNameOut', type=str)
parset.add_argument('fNameCounts', type=str)

# Get the input arguments
args = vars(parser.parse_args())	

tracemalloc.start() # Start monitoring memory
start = perf_counter() # Start timer

counts = lorentzian_histogram_numba(args['n'], bins=args['bins'])

end = perf_counter() # Stop timer
tracemalloc.stop() # Stop monitoring memory

t = end - start # Calculate time

df = pd.read_excel(args['fNameOut']) # Read in catalog

# Add entry to catalog
df.loc[-1] = [args['n'], args['bins'], args['nodes'], args['nP'], args['nP']/args['nodes'], t, tracemalloc.get_traced_memory()[1]]

# Write to the catalog
df.to_excel(args['fNameOut'], columns=["Problem Size (n)", "Bins", "Nodes", "Ranks", "Threads", "Threads Per Rank", "Runtime", "Peak Memory"])

# Save the counts to a .txt file
np.savetxt(args['fNameCounts'], np.array(counts))
