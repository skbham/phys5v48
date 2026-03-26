import time
import numpy as np

# Construct an image with repeated delta functions
deltas = np.zeros((2048, 2048))
deltas[8::16,8::16] = 1

x, y = np.meshgrid(np.linspace(-2, 2, 15), np.linspace(-2, 2, 15))
gauss = np.exp(-(x*x + y*y)/2.) / (2.*np.pi) # 2D Gaussian function

from scipy.signal import convolve2d as convolve2d_cpu

start_time = time.time()
convolved_image_using_CPU = convolve2d_cpu(deltas, gauss)
cpu_time = time.time() - start_time

print(f"Execution time: {cpu_time:g} seconds")

# Equivalent GPU code
from cupyx.profiler import benchmark
import cupy as cp
from cupyx.scipy.signal import convolve2d as convolve2d_gpu

deltas_gpu = cp.asarray(deltas)
gauss_gpu = cp.asarray(gauss)

convolved_image_using_GPU = convolve2d_gpu(deltas_gpu, gauss_gpu)
execution_gpu = benchmark(convolve2d_gpu, (deltas_gpu, gauss_gpu), n_repeat=10)
gpu_avg_time = np.average(execution_gpu.gpu_times)
print(f"{gpu_avg_time:.6f} s  (speedup = {cpu_time/gpu_avg_time:g})")

print("CPU and GPU give the same results?", np.allclose(convolved_image_using_GPU, convolved_image_using_CPU))

def transfer_compute_transferback():
    deltas_gpu = cp.asarray(deltas)
    gauss_gpu = cp.asarray(gauss)
    convolved_image_using_GPU = convolve2d_gpu(deltas_gpu, gauss_gpu)
    convolved_image_using_GPU_copied_to_host = cp.asnumpy(convolved_image_using_GPU)

execution_gpu = benchmark(transfer_compute_transferback, (), n_repeat=10)
gpu_avg_time = np.average(execution_gpu.gpu_times)
print(f"{gpu_avg_time:.6f} s  (speedup = {cpu_time/gpu_avg_time:g})  [including data transfers]")
