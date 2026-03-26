import numpy as np
import cupy, math, time
from cupyx.profiler import benchmark

def a_cpu(x, G, m, epsilon=0.01):
    """The acceleration of all masses."""
    assert epsilon > 0, "Epsilon must be greater than zero"
    dx = x - x[:,np.newaxis]  # Difference in position
    dx2 = np.sum(dx**2, axis=2)  # Distance squared
    dx2_softened = dx2 + epsilon**2  # Softened distance squared
    dx3 = dx2_softened**1.5  # Softened distance cubed
    # a_i = G * sum_{j≠i} m_j * (x_j - x_i) / |x_j - x_i|^3
    return G * np.sum(dx * (m / dx3)[:,:,np.newaxis], axis=1)

acceleration_code = r'''
extern "C" {
#define G 0.5  // Gravitational constant
#define EPS2 0.0001  // Softening parameter^2
#define WARP_SIZE 32  // Warp size for NVIDIA GPUs
#define THREADS_PER_BLOCK 256  // Threads per block for NVIDIA GPUs

__global__ void acceleration(const float *x, const float *m, float *a, const int N)
{
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;  // Pair index
    int i = i2 / N;  // Mass index i
    int j = i2 % N;  // Mass index j
    int warpId = threadIdx.x / warpSize;  // Warp index
    int laneId = threadIdx.x % warpSize;  // Thread index within warp
    __shared__ float shared_a[3 * THREADS_PER_BLOCK];  // Shared memory for accelerations

    if (i < N && j < N)  // Check if mass indices are valid
    {
        int i3 = i * 3, j3 = j * 3;  // Flat indices
        float dx[3] = {x[j3] - x[i3],
                       x[j3 + 1] - x[i3 + 1],
                       x[j3 + 2] - x[i3 + 2]};  // Difference in position
        float dx2 = dx[0] * dx[0]
                  + dx[1] * dx[1]
                  + dx[2] * dx[2];  // Distance squared
        float dx2_softened = dx2 + EPS2;  // Softened distance squared
        float dx3 = dx2_softened * sqrtf(dx2_softened);  // Softened distance cubed
        float Gm_dx3 = G * m[j] / dx3; // G * m_j / |dx|^3
        // a_i = G * sum_{j!=i} m_j * (x_j - x_i) / |x_j - x_i|^3
        int i3w = 3 * warpId * warpSize + laneId;  // Warp index in shared_a
        shared_a[i3w] += Gm_dx3 * dx[0];  // Add acceleration
        shared_a[i3w + 1] += Gm_dx3 * dx[1];  // Atomic is not needed
        shared_a[i3w + 2] += Gm_dx3 * dx[2];  // Due to warp synchronization
    }

    __syncthreads();  // Ensure all threads have finished updating shared_a

    if (laneId == 0)  // Use one thread per warp to write results back to global memory
    {
        int i3 = i * 3;  // Flat index
        int i3w = 3 * warpId * warpSize;  // Warp index in shared_a
        atomicAdd(&a[i3], shared_a[i3w]);  // Atomic add (global race condition)
        atomicAdd(&a[i3 + 1], shared_a[i3w + 1]);
        atomicAdd(&a[i3 + 2], shared_a[i3w + 2]);
    }
}
}
'''

def timestep_cpu(x0, v0, G, m, dt, epsilon=0.01):
    """Computes the next position and velocity for all masses
    given initial conditions and a time step size. """
    a0 = a_cpu(x0, G, m, epsilon)  # Initial acceleration
    v1 = a0 * dt + v0  # New velocity
    x1 = v1 * dt + x0  # New position
    return x1, v1

kick_drift_code = r'''
extern "C"
__global__ void kick_drift(const float *x0, const float *v0, const float *a0,
                           const float dt, float *x1, float *v1, const int N)
{
    int i3 = blockIdx.x * blockDim.x + threadIdx.x;  // Flat index
    if (i3 < 3 * N)  // Check if mass index is valid
    {
        v1[i3] = a0[i3] * dt + v0[i3];  // New velocity
        x1[i3] = v1[i3] * dt + x0[i3];  // New position
    }
}
'''

def initial_conditions_cpu(N, D, x_range=(0, 1), v_range=(0, 0), m_value=1.):
    """Generates initial conditions for N uniform masses with random
    starting positions and velocities in D-dimensional space."""
    np.random.seed(0)  # Set random seed for reproducibility
    x0 = np.random.uniform(*x_range, size=(N, D)).astype(np.float32)  # Random initial positions
    v0 = np.random.uniform(*v_range, size=(N, D)).astype(np.float32)  # Random initial velocities
    m = np.full(N, m_value, dtype=np.float32)  # Uniform masses
    return x0, v0, m

def initial_conditions_gpu(N, D, x_range=(0, 1), v_range=(0, 0), m_value=1.):
    """Generates initial conditions for N uniform masses with random
    starting positions and velocities in D-dimensional space."""
    cupy.random.seed(0)  # Set random seed for reproducibility
    x0 = cupy.random.uniform(*x_range, size=(N, D), dtype=cupy.float32)  # Random initial positions
    v0 = cupy.random.uniform(*v_range, size=(N, D), dtype=cupy.float32)  # Random initial velocities
    m = cupy.full(N, m_value, dtype=cupy.float32)  # Uniform masses
    return x0, v0, m

def simulate_cpu(N=512, D=3, G=0.5, m=1., dt=1e-3, t_max=1., T=None, epsilon=0.01, x_range=(0, 1), v_range=(0, 0)):
    """Simulates the motion of N masses in D-dimensional space
    under the influence of gravity for a given time period."""
    x0, v0, m = initial_conditions_cpu(N, D, x_range, v_range, m)  # Initial conditions
    if T is None:  # If T is not given
        T = int(t_max / dt)  # Number of time steps
        dt = t_max / float(T)  # Adjusted time step size
    else:
        T = int(T)  # Ensure T is an integer
        t_max = float(T) * dt  # Adjusted maximum time
    x = np.zeros([T+1, N, D])  # Positions
    v = np.zeros([T+1, N, D])  # Velocities
    x[0], v[0] = x0, v0  # Initial conditions
    for t in range(T):
        x[t+1], v[t+1] = timestep_cpu(x[t], v[t], G, m, dt, epsilon)  # Time step
    return x, v, np.linspace(0, t_max, T+1)  # Positions, velocities, and times

def simulate_gpu(N=512, D=3, m=1., dt=1e-3, t_max=1., T=None, x_range=(0, 1), v_range=(0, 0)):
    """Simulates the motion of N masses in D-dimensional space
    under the influence of gravity for a given time period."""
    # Compile the CUDA kernel
    acceleration_gpu = cupy.RawKernel(acceleration_code, "acceleration")
    kick_drift_gpu = cupy.RawKernel(kick_drift_code, "kick_drift")
    # Initial conditions
    x0, v0, m = initial_conditions_gpu(N, D, x_range, v_range, m)
    if T is None:  # If T is not given
        T = int(t_max / dt)  # Number of time steps
        dt = t_max / float(T)  # Adjusted time step size
    else:
        T = int(T)  # Ensure T is an integer
        t_max = float(T) * dt  # Adjusted maximum time
    x = cupy.zeros([T+1, N, D], dtype=cupy.float32)  # Positions
    v = cupy.zeros([T+1, N, D], dtype=cupy.float32)  # Velocities
    x[0], v[0] = x0, v0  # Initial conditions
    threads_per_block = 256  # Or 512
    assert threads_per_block % 32 == 0, "Threads per block must be a multiple of 32"
    assert threads_per_block <= 256, "Threads per block must be less than or equal to 256"
    pairs_per_grid = (N*N + threads_per_block - 1) // threads_per_block  # Round up to cover all pairs
    masses_per_grid = (N + threads_per_block - 1) // threads_per_block  # Round up to cover all masses
    for t in range(T):
        a = cupy.zeros_like(x0)  # Reset accelerations
        acceleration_gpu((pairs_per_grid,), (threads_per_block,), (x[t], m, a, N))
        kick_drift_gpu((masses_per_grid,), (threads_per_block,), (x[t], v[t], a, dt, x[t+1], v[t+1], N))
    return x, v, cupy.linspace(0, t_max, T+1)  # Positions, velocities, and times

if __name__ == "__main__":
    start = time.time()  # Start the timer
    simulate_cpu()  # N-body simulation
    stop = time.time()  # Stop the timer
    cpu_time = stop - start  # Store the runtime
    print(f"Execution time (cpu): {cpu_time:g} seconds (512 masses)")

    execution_gpu = benchmark(simulate_gpu, (), n_repeat=10)
    gpu_avg_time = np.average(execution_gpu.gpu_times)
    print(f"Execution time (gpu): {gpu_avg_time:g} seconds  ->  speedup: {cpu_time/gpu_avg_time:g}")
