import cupy

# size of the vectors
size = 2048

# allocating and populating the vectors
a_gpu = cupy.random.rand(size, dtype=cupy.float32)
b_gpu = cupy.random.rand(size, dtype=cupy.float32)
c_gpu = cupy.zeros(size, dtype=cupy.float32)
# prepare arguments
args = (a_gpu, b_gpu, c_gpu)

# CUDA code
cuda_code = r'''
extern "C" {
#define BLOCKS 2

__constant__ float factors[BLOCKS];

__global__ void sum_and_multiply(const float *A, const float *B, float *C)
{
    int item = blockIdx.x * blockDim.x + threadIdx.x;
    C[item] = (A[item] + B[item]) * factors[blockIdx.x];
}
}
'''

# compile and access the code
module = cupy.RawModule(code=cuda_code)
sum_and_multiply = module.get_function("sum_and_multiply")
# allocate and copy constant memory
factors_ptr = module.get_global("factors")
factors_gpu = cupy.ndarray(2, cupy.float32, factors_ptr)
factors_gpu[...] = cupy.random.random(2, dtype=cupy.float32)

sum_and_multiply((2, 1, 1), (size // 2, 1, 1), args)

