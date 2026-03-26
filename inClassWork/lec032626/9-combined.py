import cupy, math
import time
#from cupyx.profiler import benchmark

upper_bound = 100_000
histogram_size = 2**25

# GPU code
check_prime_gpu_code = r'''
extern "C"
__global__ void all_primes_to(int size, int * const all_prime_numbers)
{
    int number = (blockIdx.x * blockDim.x) + threadIdx.x;
    int result = 1;

    if ( number < size )
    {
        for ( int factor = 2; factor <= number / 2; factor++ )
        {
            if ( number % factor == 0 )
            {
                result = 0;
                break;
            }
        }

        all_prime_numbers[number] = result;
    }
}
'''
histogram_cuda_code = r'''
extern "C"
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ int temp_histogram[256];

    // Initialize shared memory and synchronize
    temp_histogram[threadIdx.x] = 0;
    __syncthreads();

    // Compute shared memory histogram and synchronize
    atomicAdd(&(temp_histogram[input[item]]), 1);
    __syncthreads();

    // Update global histogram
    atomicAdd(&(output[threadIdx.x]), temp_histogram[threadIdx.x]);
}
'''

# Allocate memory
all_primes_gpu = cupy.zeros(upper_bound, dtype=cupy.int32)
input_gpu = cupy.random.randint(256, size=histogram_size, dtype=cupy.int32)
output_gpu = cupy.zeros(256, dtype=cupy.int32)

# Compile and setup the grid
all_primes_to_gpu = cupy.RawKernel(check_prime_gpu_code, "all_primes_to")
grid_size_primes = (int(math.ceil(upper_bound / 1024)), 1, 1)
block_size_primes = (1024, 1, 1)
histogram_gpu = cupy.RawKernel(histogram_cuda_code, "histogram")
threads_per_block_hist = 256
grid_size_hist = (int(math.ceil(histogram_size / threads_per_block_hist)), 1, 1)
block_size_hist = (threads_per_block_hist, 1, 1)

# Execute the kernels without streams
def without_streams():
    start_time = time.time()
    all_primes_to_gpu(grid_size_primes, block_size_primes, (upper_bound, all_primes_gpu))
    histogram_gpu(grid_size_hist, block_size_hist, (input_gpu, output_gpu))

    # Save results
    output_one = all_primes_gpu
    output_two = output_gpu
    return time.time() - start_time

# Execute the kernels with streams
def with_streams():
    start_time = time.time()
    stream_one = cupy.cuda.Stream()
    stream_two = cupy.cuda.Stream()

    with stream_one:
        all_primes_to_gpu(grid_size_primes, block_size_primes, (upper_bound, all_primes_gpu))
    with stream_two:
        histogram_gpu(grid_size_hist, block_size_hist, (input_gpu, output_gpu))

    # Save results
    output_one = all_primes_gpu
    output_two = output_gpu
    return time.time() - start_time

# Execute the kernels with a CUDA event
def with_event():
    start_time = time.time()
    stream_one = cupy.cuda.Stream()
    stream_two = cupy.cuda.Stream()
    sync_point = cupy.cuda.Event()

    with stream_one:
        all_primes_to_gpu(grid_size_primes, block_size_primes, (upper_bound, all_primes_gpu))
        sync_point.record(stream=stream_one) # Record event of first half finishing
        all_primes_to_gpu(grid_size_primes, block_size_primes, (upper_bound, all_primes_gpu))
    with stream_two:
        stream_two.wait_event(sync_point) # Don't start until first half of other stream finishes
        histogram_gpu(grid_size_hist, block_size_hist, (input_gpu, output_gpu))

    # Save results
    output_one = all_primes_gpu
    output_two = output_gpu
    return time.time() - start_time

time_without_streams = without_streams()
time_with_streams = with_streams()
time_with_events = with_event()

print(f"time without streams: {time_without_streams:g} seconds")
print(f"time with streams: {time_with_streams:g} seconds")
print(f"time with events: {time_with_events:g} seconds")
print(f"(without streams) / (with streams) = {time_without_streams/time_with_streams:g}")
print(f"(with streams) / (with events) = {time_with_streams/time_with_events:g}")
