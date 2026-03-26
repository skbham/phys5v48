import math, cupy, time
import numpy as np

# size of the vectors
size = 100_000_000

# allocating and populating the vectors
a_gpu = cupy.random.rand(size, dtype=cupy.float32)
b_gpu = cupy.random.rand(size, dtype=cupy.float32)
c_gpu = cupy.zeros(size, dtype=cupy.float32)
a_cpu = cupy.asnumpy(a_gpu)
b_cpu = cupy.asnumpy(b_gpu)
c_cpu = np.zeros(size, dtype=np.float32)

# CPU code
def vector_add(A, B, C, size):
    for item in range(0, size):
        C[item] = A[item] + B[item]

# CUDA vector_add
vector_add_gpu = cupy.RawKernel(r'''
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
   int item = (blockIdx.x * blockDim.x) + threadIdx.x;
   if ( item < size )
   {
      C[item] = A[item] + B[item];
   }
}
''', "vector_add")

# execute the code and measure time
#%timeit -n 1 -r 1 vector_add(a_cpu, b_cpu, c_cpu, size)
start_time = time.time()
vector_add(a_cpu, b_cpu, c_cpu, size)
cpu_time = time.time() - start_time
print(f"Execution time (cpu): {cpu_time:g} seconds")

threads_per_block = 1024
grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
block_size = (threads_per_block, 1, 1)
gpu_times = []
for _ in range(10):
    start_gpu = cupy.cuda.Event() # Timer events
    end_gpu = cupy.cuda.Event()
    start_gpu.record()  # Start!
    vector_add_gpu(grid_size, block_size, (a_gpu, b_gpu, c_gpu, size))
    end_gpu.record()  # Stop!
    end_gpu.synchronize()  # Double stop!
    gpu_times.append(cupy.cuda.get_elapsed_time(start_gpu, end_gpu))
gpu_avg_time = np.average(gpu_times)
print(f"Execution time (gpu): {gpu_avg_time:g} seconds  ->  speedup: {cpu_time/gpu_avg_time:g}")

# test
if np.allclose(c_cpu, c_gpu):
    print("Correct results!")
else:
    print("Wrong results!")
