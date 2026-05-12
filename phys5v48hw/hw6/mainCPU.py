
# PHYS 5V48: Homework 6
# Implement Forward Euler, RK2 (midpoint), and RK4
# on both CPU and GPU using NumPy and CuPy.

# Explicit Runge-Kutta methods
# Implicit TRBDF2 method

#from calendar import c
from copy import Error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import perf_counter

from pandas.core.dtypes.cast import NumpyArrayT
from pandas.core.series import nargsort

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
    return - y

def dydt2(t, y):
    r = 1
    return r * y * (1 - y)

def dydt3(t, y):
    alpha = 1
    return - alpha * y

# RK2
def rk2(y, dydt, h=0.01):

    tn = 0
    yn = y(tn)

    stepNum = int(1/h)

    for i in range(0, stepNum):

        # For the Midpoint Method set:
        a1 = 0
        a2 = 1
        p1 = 1/2
        q11 = 1/2

        k1 = dydt(tn, yn)
        k2 = dydt(tn + (np.pi * h), yn + (q11 * k1 * h))

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

    nArr = np.array([4,5,6,7,8,9,10])
    hArr = np.array(2.0 ** (-nArr))
    nNum = len(nArr)

    yRk2CpuArr = np.zeros(nNum)
    yRk4CpuArr = np.zeros(nNum)
    yForEulerCpuArr = np.zeros(nNum)
    globErrRk2CpuArr = np.zeros(nNum)
    globErrRk4CpuArr = np.zeros(nNum) 
    globErrForEulerCpuArr = np.zeros(nNum)
    
    tRk2Cpu = np.zeros(nNum)
    tRk4Cpu = np.zeros(nNum)
    tForEulerCpu = np.zeros(nNum)

    fname = "hw6CPUData.xlsx"

    # Run CPU implementations

    # Run RK2 on CPU
    for i in range(0, nNum):

        start = perf_counter() # Start timer
        yRk2CpuArr[i] = rk2(y, dydt, h=hArr[i])
        end = perf_counter() # Stop timer
        tRk2Cpu[i] = end - start # Calculate time

        globErrRk2CpuArr[i] = np.abs(yRk2CpuArr[i] - np.exp(-1))

    #print("print(tRk2Cpu): " + str(tRk2Cpu))

    # Run RK4 on CPU
    for i in range(0, nNum):

        start = perf_counter() # Start timer
        yRk4CpuArr[i] = rk4(y, dydt, h=hArr[i])
        end = perf_counter() # Stop timer
        tRk4Cpu[i] = end - start # Calculate time

        globErrRk4CpuArr[i] = np.abs(yRk4CpuArr[i] - np.exp(-1))

    #print("print(tRk4Cpu): " + str(tRk4Cpu))

    # Run Forward Euler on CPU
    for i in range(0, nNum):

        start = perf_counter() # Start timer
        yForEulerCpuArr[i] = forEuler(y, dydt, h=hArr[i])
        end = perf_counter() # Stop timer
        tForEulerCpu[i] = end - start # Calculate time

        globErrForEulerCpuArr[i] = np.abs(yForEulerCpuArr[i] - np.exp(-1))

    #print("print(tForEulerCpu): " + str(tForEulerCpu))

    # Plot CPU graphs

    # Error plots
    plt.plot(hArr, globErrRk2CpuArr, label="RK2")
    plt.plot(hArr, globErrRk4CpuArr, label="RK4")
    plt.plot(hArr, globErrForEulerCpuArr, label="Forward Euler")

    plt.xscale("log")
    plt.yscale("log")

    plt.title("Error Vs. Step Size")
    plt.xlabel("Step Size")
    plt.ylabel("Error")

    plt.legend()

    plt.savefig("cpuPlots1.png")
    plt.close()

    # Runtime plots
    plt.plot(hArr, tRk2Cpu, label="RK2")
    plt.plot(hArr, tRk4Cpu, label="RK4")
    plt.plot(hArr, tForEulerCpu, label="Forward Euler")

    plt.xscale("log")
    plt.yscale("log")

    plt.title("Runtime Vs. Step Size")
    plt.xlabel("Step Size")
    plt.ylabel("Runtime")

    plt.legend()

    plt.savefig("cpuPlots2.png")

    writer = pd.ExcelWriter(fname, engine='openpyxl', mode='a')
    #df = pd.read_excel(writer, index_col=0) # Read in catalog

    # Write to the catalog
    colList = ["Method", "n", "CPU Global Error", "CPU Time (s)"]
    
    labRk2 = np.full(nNum, "RK2")
    labRk4 = np.full(nNum, "RK4")
    labForEuler = np.full(nNum, "Forward Euler")

    rowNames = np.concatenate((labRk2, labRk4, labForEuler))
    errors = np.concatenate((globErrRk2CpuArr, globErrRk4CpuArr, globErrForEulerCpuArr))
    times = np.concatenate((tRk2Cpu, tRk4Cpu, tForEulerCpu))
    nTot = np.concatenate((nArr, nArr, nArr))
    #gpuTimes = np.array([tRk2Gpu, tRk4Gpu, tForEulerGpu])
    #speedup = cpuTimes / gpuTimes

    data = np.stack((rowNames, nTot, errors, times), axis=-1)
    print(data)

    df = pd.DataFrame(data,columns=colList)
    print(df)

    df.to_excel(excel_writer=fname, columns=colList, engine='openpyxl')

    return 0

main()

# Run a convergence study
# deltat = 2 ** (-n)
# n in [4,5,6,7,8,9,10]
# errorGlobal = np.abs(y(1) - np.exp(-1)) # at t=1

