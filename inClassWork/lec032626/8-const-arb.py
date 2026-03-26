import cupy, math

# size of the vectors
size = 10**6

# allocating and populating the vectors
a_gpu = cupy.random.rand(size, dtype=cupy.float32)
b_gpu = cupy.random.rand(size, dtype=cupy.float32)
c_gpu = cupy.zeros(size, dtype=cupy.float32)
# prepare arguments
args = (a_gpu, b_gpu, c_gpu, size)

# CUDA code
cuda_code = r'''
extern "C" {
__constant__ float factors[BLOCKS];

__global__ void sum_and_multiply(const float *A, const float *B, float *C, const int size)
{
    int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item < size)
        C[item] = (A[item] + B[item]) * factors[blockIdx.x];
}
}
'''

# compute the number of blocks and replace "BLOCKS" in the CUDA code
threads_per_block = 1024
num_blocks = int(math.ceil(size / threads_per_block))
cuda_code = cuda_code.replace("BLOCKS", f"{num_blocks}") 

# compile and access the code
module = cupy.RawModule(code=cuda_code)
sum_and_multiply = module.get_function("sum_and_multiply")
# allocate and copy constant memory
factors_ptr = module.get_global("factors")
factors_gpu = cupy.ndarray(num_blocks, cupy.float32, factors_ptr)
factors_gpu[...] = cupy.random.random(num_blocks, dtype=cupy.float32)

sum_and_multiply((num_blocks, 1, 1), (threads_per_block, 1, 1), args)

