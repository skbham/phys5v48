
# PHYS 5V48: Homework 6
# Implement Forward Euler, RK2 (midpoint), and RK4
# on both CPU and GPU using NumPy and CuPy.

# Explicit Runge-Kutta methods
# Implicit TRBDF2 method

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import perf_counter

def y(t):
    return np.exp(-t)

def dydt(t, y):
    return - y

def dydt2(t, y):
    r = 2
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

    yRk2Arr = np.zeros(nNum)
    yRk4Arr = np.zeros(nNum)
    yForEulerArr = np.zeros(nNum)
    globErrRk2Arr = np.zeros(nNum)
    globErrRk4Arr = np.zeros(nNum) 
    globErrForEulerArr = np.zeros(nNum)
    
    tRk2 = np.zeros(nNum)
    tRk4 = np.zeros(nNum)
    tForEuler = np.zeros(nNum)

    fname = "hw6CPUData.xlsx"

    # Run CPU implementations

    # Run RK2 on CPU
    for i in range(0, nNum):

        start = perf_counter() # Start timer
        yRk2Arr[i] = rk2(y, dydt, h=hArr[i])
        end = perf_counter() # Stop timer
        tRk2[i] = end - start # Calculate time

        globErrRk2Arr[i] = np.abs(yRk2Arr[i] - np.exp(-1))

    #print("print(tRk2Cpu): " + str(tRk2Cpu))

    # Run RK4 on CPU
    for i in range(0, nNum):

        start = perf_counter() # Start timer
        yRk4Arr[i] = rk4(y, dydt, h=hArr[i])
        end = perf_counter() # Stop timer
        tRk4[i] = end - start # Calculate time

        globErrRk4Arr[i] = np.abs(yRk4Arr[i] - np.exp(-1))

    #print("print(tRk4Cpu): " + str(tRk4Cpu))

    # Run Forward Euler on CPU
    for i in range(0, nNum):

        start = perf_counter() # Start timer
        yForEulerArr[i] = forEuler(y, dydt, h=hArr[i])
        end = perf_counter() # Stop timer
        tForEuler[i] = end - start # Calculate time

        globErrForEulerArr[i] = np.abs(yForEulerArr[i] - np.exp(-1))

    #print("print(tForEulerCpu): " + str(tForEulerCpu))

    # Plot CPU graphs

    # Error plots
    plt.plot(hArr, globErrRk2Arr, label="RK2")
    plt.plot(hArr, globErrRk4Arr, label="RK4")
    plt.plot(hArr, globErrForEulerArr, label="Forward Euler")

    plt.xscale("log")
    plt.yscale("log")

    plt.title("Error Vs. Step Size")
    plt.xlabel("Step Size")
    plt.ylabel("Error")

    plt.legend()

    plt.savefig("cpuPlots1.png")
    plt.close()

    # Runtime plots
    plt.plot(hArr, tRk2, label="RK2")
    plt.plot(hArr, tRk4, label="RK4")
    plt.plot(hArr, tForEuler, label="Forward Euler")

    plt.xscale("log")
    plt.yscale("log")

    plt.title("Runtime Vs. Step Size")
    plt.xlabel("Step Size")
    plt.ylabel("Runtime (s)")

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
    errors = np.concatenate((globErrRk2Arr, globErrRk4Arr, globErrForEulerArr))
    times = np.concatenate((tRk2, tRk4, tForEuler))
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

