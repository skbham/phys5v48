# CPU Function
def find_all_primes_cpu(upper):
    all_prime_numbers = []
    for num in range(0, upper):
        prime = True
        for i in range(2, (num // 2) + 1):
            if (num % i) == 0:
                prime = False
                break
        if prime:
            all_prime_numbers.append(num)
    return all_prime_numbers

# GPU Kernel
from numba import cuda

@cuda.jit
def check_prime_gpu_kernel(num, result):
   result[0] = num
   for i in range(2, (num // 2) + 1):
       if (num % i) == 0:
           result[0] = 0
           break

# Example use of GPU Kernel
import time
import numpy as np

result = np.zeros((1), np.int32)
check_prime_gpu_kernel[1, 1](11, result)
print(result[0])
check_prime_gpu_kernel[1, 1](12, result)
print(result[0])

def find_all_primes_cpu_and_gpu(upper):
    all_prime_numbers = []
    for num in range(0, upper):
        result = np.zeros((1), np.int32)
        check_prime_gpu_kernel[1,1](num, result)
        if result[0] != 0:
            all_prime_numbers.append(num)
    return all_prime_numbers

start_time = time.time()
find_all_primes_cpu(100_000)
cpu_time = time.time() - start_time
print(f"Execution time (cpu): {cpu_time:g} seconds (10,000 primes)")

start_time = time.time()
find_all_primes_cpu_and_gpu(10_000)
kernel_time = time.time() - start_time
print(f"Execution time (cpu+gpu): {kernel_time:g} seconds (100,000 primes)")

# Numba GPU Device Offloading
import numba as nb

@nb.vectorize(['int32(int32)'], target='cuda')
def check_prime_gpu(num):
    for i in range(2, (num // 2) + 1):
       if (num % i) == 0:
           return 0
    return num

start_time = time.time()
numbers = np.arange(0, 100_000, dtype=np.int32)
check_prime_gpu(numbers)
gpu_time = time.time() - start_time
print(f"Execution time (gpu): {gpu_time:g} seconds  ->  speedup: {10.*cpu_time/gpu_time:g}")
