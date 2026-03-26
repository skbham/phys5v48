# Python vector_add
def vector_add(A, B, C):
    for i in range(len(A)):
        C[i] = A[i] + B[i]

# CUDA vector_add
vector_add_cuda_code = r'''
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}
'''

import cupy

# size of the vectors
size = 1024
#size = 2048 # Will not work! 1024 threads max

# allocating and populating the vectors
a_gpu = cupy.random.rand(size, dtype=cupy.float32)
b_gpu = cupy.random.rand(size, dtype=cupy.float32)
c_gpu = cupy.zeros(size, dtype=cupy.float32)

vector_add_gpu = cupy.RawKernel(vector_add_cuda_code, "vector_add")

vector_add_gpu((1, 1, 1), (size, 1, 1), (a_gpu, b_gpu, c_gpu))

import numpy as np

a_cpu = cupy.asnumpy(a_gpu)
b_cpu = cupy.asnumpy(b_gpu)
c_cpu = np.zeros(size, dtype=np.float32)

vector_add(a_cpu, b_cpu, c_cpu)

# test
if np.allclose(c_cpu, c_gpu):
    print("Correct results!")
