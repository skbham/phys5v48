import math
import numpy as np
import cupy
import time
from cupyx.profiler import benchmark

def histogram(input_array, output_array):
    for item in input_array:
        output_array[item] += 1

# input size
size = 2**25

# allocate memory on CPU and GPU
input_gpu = cupy.random.randint(256, size=size, dtype=cupy.int32)
input_cpu = cupy.asnumpy(input_gpu)
output_gpu = cupy.zeros(256, dtype=cupy.int32)
output_cpu = cupy.asnumpy(output_gpu)

# CUDA code
histogram_cuda_code = r'''
extern "C"
__global__ void histogram(const int *input, int *output)
{
    int item = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int temp_histogram[256];

    // Initialize shared memory and synchronize
    temp_histogram[threadIdx.x] = 0;
    __syncthreads();

    // Compute shared memory histogram and synchronize
    atomicAdd(&temp_histogram[input[item]], 1);
    __syncthreads();

    // Update global histogram
    atomicAdd(&output[threadIdx.x], temp_histogram[threadIdx.x]);
}
'''

# compile and setup CUDA code
histogram_gpu = cupy.RawKernel(histogram_cuda_code, "histogram")
threads_per_block = 256
grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
block_size = (threads_per_block, 1, 1)

# check correctness
histogram(input_cpu, output_cpu)
histogram_gpu(grid_size, block_size, (input_gpu, output_gpu))
if np.allclose(output_cpu, output_gpu):
    print("Correct results!")
else:
    print("Wrong results!")

# measure performance
# %timeit -n 1 -r 1 histogram(input_cpu, output_cpu)
start_time = time.time()
histogram(input_cpu, output_cpu)
cpu_time = time.time() - start_time
print(f"Execution time (cpu): {cpu_time:g} seconds")

execution_gpu = benchmark(histogram_gpu, (grid_size, block_size, (input_gpu, output_gpu)), n_repeat=10)
gpu_avg_time = np.average(execution_gpu.gpu_times)
print(f"Execution time (gpu): {gpu_avg_time:g} seconds  ->  speedup: {cpu_time/gpu_avg_time:g}")
