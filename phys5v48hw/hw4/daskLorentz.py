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

counts = run_dask(args['n'])

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
