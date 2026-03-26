import math
import numpy as np
import cupy

# vector size
size = 2048

# GPU memory allocation
a_gpu = cupy.random.rand(size, dtype=cupy.float32)
b_gpu = cupy.random.rand(size, dtype=cupy.float32)
c_gpu = cupy.zeros(size, dtype=cupy.float32)
gpu_args = (a_gpu, b_gpu, c_gpu, size)

# CPU memory allocation
a_cpu = cupy.asnumpy(a_gpu)
b_cpu = cupy.asnumpy(b_gpu)
c_cpu = np.zeros(size, dtype=np.float32)

# Python code
def vector_add(A, B, C):
    for i in range(len(A)):
        C[i] = A[i] + B[i]

# CUDA code
vector_add_cuda_code = r'''
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
  int item = (blockIdx.x * blockDim.x) + threadIdx.x;
  int offset = threadIdx.x * 3;
  extern __shared__ float temp[];

  if ( item < size )
  {
      temp[offset + 0] = A[item];
      temp[offset + 1] = B[item];
      temp[offset + 2] = temp[offset + 0] + temp[offset + 1];
      C[item] = temp[offset + 2];
  }
}
'''

# compile and execute code
vector_add_gpu = cupy.RawKernel(vector_add_cuda_code, "vector_add")
threads_per_block = 32
grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
block_size = (threads_per_block, 1, 1)
mem_size = threads_per_block * 3 * cupy.dtype(cupy.float32).itemsize
vector_add_gpu(grid_size, block_size, gpu_args, shared_mem=(mem_size))

# execute Python code and compare results
vector_add(a_cpu, b_cpu, c_cpu)
result = np.allclose(c_cpu, c_gpu)
print(f'CPU and GPU agree? {result}')
