""" MA3.py

Student:
Mail:
Reviewed by:
Date reviewed:

"""
import random
import matplotlib.pyplot as plt
import math as m
import numpy as np
import time 
import concurrent.futures as future
from statistics import mean 
from time import perf_counter as pc
from numba import njit 
from multiprocessing import Process, cpu_count
from itertools import repeat 

# Exc1
def approximate_pi(n):
    
    num_points = n
    inside_circle = 0 

    inside_x, inside_y = [], []
    outside_x, outside_y = [], []

    for i in range(num_points):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)

        if (x**2 + y**2) <= 1:
            inside_circle += 1 
            inside_x.append(x)
            inside_y.append(y)
        else:
            outside_x.append(x)
            outside_y.append(y)

    pi_estimate = 4 * (inside_circle / num_points)
    
    #print(f"Antal punkter: {num_points}")
    print(f"Approximation av pi: {pi_estimate:.6f}")

    # Rita punkterna: röda innanför cirkeln, blå utanför
    plt.figure(figsize=(6, 6))
    plt.scatter(inside_x, inside_y, color="red", s=2, label="Inne i cirkeln")
    plt.scatter(outside_x, outside_y, color="blue", s=2, label="Utanför cirkeln")

    plt.savefig(f"pi_approx_{n}.png")
    plt.close()

    return pi_estimate


# Exc2, approximation
def sphere_volume(n, d): #n=total amount of points, d=dimensions

    points = ([random.uniform(-1, 1) for _ in range(d)] for _ in range (n)) #för varje punkt, så skapas en lista med ett random nummer för varje dimension

    sum_squared = list(map(lambda pointlist: sum([x**2 for x in pointlist]), points )) #för varje punkt i listan så skapas en ny lista där varje punkt har kvadrerats
    
    inside = list(filter(lambda s: s <= 1, sum_squared)) #varje punkt i sum_squared gås igenom, där punkterna som är mindre eller lika med 1 läggs till i en ny lista 

    volume = (2**d) * (len(inside) / n ) #V_sphere = V_cube * (points in the sphere/total amount of points)

    return volume

#Exc2, real value
def hypersphere_exact(n, d):

    V_d = (np.pi**(d/2)) / (m.gamma((d/2) + 1))
    return V_d


#Exc 3
@njit #gör om numeriska funktioner till snabb maskin-kod 
def sphere_volume_numba(n:int, d:int)->float:
    inside_count = 0

    for i in range(n):
        

        squared_sum = 0

        for j in range(d):
            x = random.uniform(-1, 1)
            squared_sum += x**2

        if squared_sum <= 1:
            inside_count += 1 

    return (2**d) * (inside_count / n ) #V_sphere = V_cube * points in the sphere/total amount of points 
    


#Exc4: parallel code - parallelize actual computations by splitting data
def sphere_volume_parallel(n, d, np=10):
    points_per_process = n // np
    "skapa en grupp workers som kan ta sphere_volume - beräkningar"
   
    with future.ProcessPoolExecutor(max_workers=np) as ex:

        for i in range(np):
             p = ex.submit(sphere_volume, points_per_process, d)
             r = p.result()
             return r

        return mean(r)



def main():
    # Exc1
    dots = [1000, 10000, 100000]
    for n in dots:
        approximate_pi(n)

    # Exc2
    n = 100000
    d = 2
    sphere_volume(n, d)
    print(f"Approximated volume of {d} dimensional sphere = {sphere_volume(n, d)}")
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    n = 100000
    d = 11
    hypersphere_exact(n, d)
    print(f"Approximated volume of {d} dimensional sphere = {sphere_volume(n, d)}")
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    # Exc3
    n = 1000000
    d = 11
    
    for i in range(3):
        start = pc()
        sphere_volume_numba(n, d)
        stop = pc()
        print(f"Exc3: Sequential time of {d} and {n}: {stop-start}")
    
    for i in range(3):
        start = pc()
        sphere_volume_numba(n, d)
        stop = pc()
        print(f"Exc3: Numba time: {stop-start}")

    # Exc4
    n = 1000000
    d = 11
    for i in range(3):
        start = pc()
        sphere_volume(n, d)
        stop = pc()

        print(f"Exc4: Sequential time of {d} and {n}: {stop-start}")

    for i in range(3):
        start = pc()
        sphere_volume_parallel(n, d)
        stop = pc()

        print(f"Exc4: Parallel time: {stop-start}")

    
    

if __name__ == '__main__':
	main()
