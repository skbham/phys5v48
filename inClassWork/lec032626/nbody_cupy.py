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

def a_gpu(x, G, m, epsilon=0.01):
    """The acceleration of all masses."""
    assert epsilon > 0, "Epsilon must be greater than zero"
    dx = x - x[:,cupy.newaxis]  # Difference in position
    dx2 = cupy.sum(dx**2, axis=2)  # Distance squared
    dx2_softened = dx2 + epsilon**2  # Softened distance squared
    dx3 = dx2_softened**1.5  # Softened distance cubed
    # a_i = G * sum_{j≠i} m_j * (x_j - x_i) / |x_j - x_i|^3
    return G * cupy.sum(dx * (m / dx3)[:,:,cupy.newaxis], axis=1)

def timestep_cpu(x0, v0, G, m, dt, epsilon=0.01):
    """Computes the next position and velocity for all masses
    given initial conditions and a time step size. """
    a0 = a_cpu(x0, G, m, epsilon)  # Initial acceleration
    v1 = a0 * dt + v0  # New velocity
    x1 = v1 * dt + x0  # New position
    return x1, v1

def timestep_gpu(x0, v0, G, m, dt, epsilon=0.01):
    """Computes the next position and velocity for all masses
    given initial conditions and a time step size. """
    a0 = a_gpu(x0, G, m, epsilon)  # Initial acceleration
    v1 = a0 * dt + v0  # New velocity
    x1 = v1 * dt + x0  # New position
    return x1, v1

def initial_conditions_cpu(N, D, x_range=(0, 1), v_range=(0, 0), m_value=1.):
    """Generates initial conditions for N uniform masses with random
    starting positions and velocities in D-dimensional space."""
    np.random.seed(0)  # Set random seed for reproducibility
    x0 = np.random.uniform(*x_range, size=(N, D))  # Random initial positions
    v0 = np.random.uniform(*v_range, size=(N, D))  # Random initial velocities
    m = np.full(N, m_value, dtype=np.float64)  # Uniform masses
    return x0, v0, m

def initial_conditions_gpu(N, D, x_range=(0, 1), v_range=(0, 0), m_value=1.):
    """Generates initial conditions for N uniform masses with random
    starting positions and velocities in D-dimensional space."""
    cupy.random.seed(0)  # Set random seed for reproducibility
    x0 = cupy.random.uniform(*x_range, size=(N, D))  # Random initial positions
    v0 = cupy.random.uniform(*v_range, size=(N, D))  # Random initial velocities
    m = cupy.full(N, m_value, dtype=cupy.float64)  # Uniform masses
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

def simulate_gpu(N=512, D=3, G=0.5, m=1., dt=1e-3, t_max=1., T=None, epsilon=0.01, x_range=(0, 1), v_range=(0, 0)):
    """Simulates the motion of N masses in D-dimensional space
    under the influence of gravity for a given time period."""
    x0, v0, m = initial_conditions_gpu(N, D, x_range, v_range, m)  # Initial conditions
    if T is None:  # If T is not given
        T = int(t_max / dt)  # Number of time steps
        dt = t_max / float(T)  # Adjusted time step size
    else:
        T = int(T)  # Ensure T is an integer
        t_max = float(T) * dt  # Adjusted maximum time
    x = cupy.zeros([T+1, N, D])  # Positions
    v = cupy.zeros([T+1, N, D])  # Velocities
    x[0], v[0] = x0, v0  # Initial conditions
    for t in range(T):
        x[t+1], v[t+1] = timestep_gpu(x[t], v[t], G, m, dt, epsilon)  # Time step
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
