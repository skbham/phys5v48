
import argparse
import asyncio
import numpy as np
import openpyxl
import os
import pandas as pd
from time import perf_counter
import tracemalloc
import resource

# import custom modules
import asyncLorentz # Import the set of functions
import threadLorentz
import mpLorentz

# Initialize the parser
parser = argparse.ArgumentParser()

# Add arguments to the parser
parser.add_argument('n', type=int)
parser.add_argument('bins', type=int)
parser.add_argument('nP', type=int)
parser.add_argument('nodes', type=int)
parser.add_argument('fNameOut', type=str)
parser.add_argument('fNameCounts', type=str)

# Get the input arguments
args = vars(parser.parse_args())	

tracemalloc.start() # Start monitoring memory
start = perf_counter() # Start timer

#counts = asyncLorentz.run_async(args['n'], n_tasks=args['nP'], bins=args['bins'], n_subchunks=10) # AsyncIO
#counts = threadLorentz.run_threaded(args['n'], n_threads=args['nP'], bins=args['bins']) # Threading
counts = mpLorentz.run_multiproc(args['n'], n_threads=args['nP'], bins=args['bins']) # Threading

end = perf_counter() # Stop timer
tracemalloc.stop() # Stop monitoring memory

t = end - start # Calculate time

colList = ["Problem Size (n)", "Bins", "Nodes", "Ranks", "Threads Per Rank", "Runtime", "Peak Memory"]

writer = pd.ExcelWriter(args['fNameOut'], engine='openpyxl', mode='a')
df = pd.read_excel(writer, index_col=0) # Read in catalog

# Add entry to catalog

#peakMem = tracemalloc.get_traced_memory()[1]
peakMem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

# newRow = [args['n'], args['bins'], args['nodes'], args['nP'], 10, t, peakMem] # AsyncIO
newRow = [args['n'], args['bins'], args['nodes'], args['nP'], 1, t, peakMem] # Threading

df.loc[len(df)] = newRow

# Write to the catalog
df.to_excel(excel_writer=args['fNameOut'], columns=colList, engine='openpyxl')

# Save the counts to a .txt file
np.savetxt(args['fNameCounts'], np.array(counts))


