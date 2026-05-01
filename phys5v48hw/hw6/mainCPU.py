
# PHYS 5V48: Homework 6
# Implement Forward Euler, RK2 (midpoint), and RK4
# on both CPU and GPU using NumPy and CuPy.

# Explicit Runge-Kutta methods
# Implicit TRBDF2 method

from calendar import c
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter

# dudt = f(t,u)
# t in [t0, tf]
# u(t0) = u0

# dydt = -y
# t in [0,1]
# y(0) = 1
# y(t) = e^(-t)

def y(t):
    return np.exp(-t)

def dydt(t, y):
    return - y(t)

def dydt2(t, y):
    r = 1
    return r * y(t) * (1 - y(t))

def dydt3(t, y):
    alpha = 1
    return - alpha * y(t)

# RK2
def rk2(y, dydt, h=0.01):

    tn = 0
    yn = y(tn)

    stepNum = int(1/h)

    for i in range(0, stepNum):

        # For the Midpoint Method set:
        a1 = 0
        p1 = 1/2
        q11 = 1/2

        k1 = dydt(tn, yn)
        k2 = dydt(tn + (pi * h), yn + (q11 * k1 * h))

        yn = yn + ((a1 * k1) + (a2 * k2)) * h

        tn = tn + h

    return yn

# RK4
def rk4(y, dydt, h=0.01):

    tn = 0
    yn = y(tn)

    stepNum = int(1/h)

    for i in range(0,stepNum):

        k1 = h * dydt(tn, yn)
        k2 = h * dydt(tn + (h/2), yn + (k1/2))
        k3 = h * dydt(tn + (h/2), yn + (k2/2))
        k4 = h * dydt(tn + h, yn + k3)

        yn = yn + (k1/6) + (k2/3) + (k3/3) + (k4/6)

        tn = tn + h
    
    return yn

# Forward Euler
def forEuler(y, dydt, h=0.01):

    tn = 0
    yn = y(tn)

    stepNum = int(1/h)

    for i in range(0,stepNum):

        yn = yn + h * dydt(tn, yn)
        tn = tn + h

    return yn

def main():

    nArr = np.array([4,5])
    hArr = np.array(2.0 ** (-nArr))
    nNum = 2

    yRk2CpuArr = np.zeros(nNum)
    yRk4CpuArr = np.zeros(nNum)
    yForEulerCpuArr = np.zeros(nNum)
    globErrRk2CpuArr = np.zeros(nNum)
    globErrRk4CpuArr = np.zeros(nNum) 
    globErrForEulerCpuArr = np.zeros(nNum)
    
    fname = "hw6CPUData.xlsx"

    # Run CPU implementations

    # Run RK2 on CPU
    start = perf_counter() # Start timer
    for i in range(0, nNum):

        yRk2CpuArr[i] = rk2(y, dydt, h=hArr[i])
        globErrRk2CpuArr[i] = np.abs(yRk2CpuArr[i] - np.exp(-1))

    end = perf_counter() # Stop timer
    tRk2Cpu = end - start # Calculate time

    # Run RK4 on CPU
    start = perf_counter() # Start timer
    for i in range(0, nNum):

        yRk4CpuArr[i] = rk4(y, dydt, h=hArr[i])
        globErrRk4CpuArr[i] = np.abs(yRk4CpuArr[i] - np.exp(-1))

    end = perf_counter() # Stop timer
    tRk4Cpu = end - start # Calculate time

    # Run Forward Euler on CPU
    start = perf_counter() # Start timer
    for i in range(0, nNum):

        yForEulerCpuArr[i] = forEuler(y, dydt, h=hArr[i])
        globErrForEulerCpuArr[i] = np.abs(yForEulerCpuArr[i] - np.exp(-1))

    end = perf_counter() # Stop timer
    tForEulerCpu = end - start # Calculate time

    # Plot CPU graphs
    #plt.figure()
    plt.plot(globErrRk2CpuArr, hArr)
    plt.plot(globErrRk4CpuArr, hArr)
    plt.plot(globErrForEulerCpuArr, hArr)

    plt.savefig()

    # Plot GPU graphs
    #plt.figure()
    plt.plot(globErrRk2CpuArr, hArr)
    plt.plot(globErrRk4CpuArr, hArr)
    plt.plot(globErrForEulerCpuArr, hArr)

    plt.savefig()

    writer = pd.ExcelWriter(fname, engine='openpyxl', mode='a')
    #df = pd.read_excel(writer, index_col=0) # Read in catalog

    # Write to the catalog
    colList = ["Method", "CPU Time (s)", "GPU Time (s)", "Speedup (CPU/GPU)"]
    rowNames = np.array(["RK2", "RK4", "Forward Euler"])
    cpuTimes = np.array([tRk2Cpu, tRk4Cpu, tForEulerCpu])
    #gpuTimes = np.array([tRk2Gpu, tRk4Gpu, tForEulerGpu])
    #speedup = cpuTimes / gpuTimes

    data = np.concatenate(rowName, cpuTimes)
    print(data)

    df = pd.DataFrame(data)

    df.to_excel(excel_writer=fname, columns=colList, engine='openpyxl')

    return 0

main()

# Run a convergence study
# deltat = 2 ** (-n)
# n in [4,5,6,7,8,9,10]
# errorGlobal = np.abs(y(1) - np.exp(-1)) # at t=1

