# GPU Speedup Test

import time
import numpy as np

# Set up large list of random numbers
size = 4096**2
input = np.random.random(size).astype(np.float32)

# Time how long it takes to sort
# %timeit -n 1 -r 1 output = np.sort(input)
start_time = time.time()
output = np.sort(input)
cpu_time = time.time() - start_time

print(f"Execution time: {cpu_time:g} seconds")

# Equivalent GPU code
from cupyx.profiler import benchmark
import cupy as cp

input_gpu = cp.asarray(input)
execution_gpu = benchmark(cp.sort, (input_gpu,), n_repeat=10)
gpu_avg_time = np.average(execution_gpu.gpu_times)
print(f"{gpu_avg_time:.6f} s  (speedup = {cpu_time/gpu_avg_time:g})")
