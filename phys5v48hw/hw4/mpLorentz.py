# mp_lorentz.py
import multiprocessing
def run_multiproc(n, n_cores=4, bins=100, xmin=-10, xmax=10):
    """
    Run the Lorentzian sampling in parallel using processes.
    """
    # Split n samples among processes
    chunks = (n // n_cores) * np.ones(n_cores, dtype=int)
    chunks[:n % n_cores] += 1 # Distribute remainder
    # Use partial function to reset default arguments (bins, xmin, xmax)
    from functools import partial
    lorentzian_hist_func = partial(lorentzian_histogram, bins=bins, xmin=xmin, xmax=xmax)
    # Use Pool to distribute chunks to processes
    with multiprocessing.Pool(n_cores) as pool:
        results = pool.map(lorentzian_hist_func, chunks)
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

counts = run_multiproc(args['n'], bins=args['bins'])

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
