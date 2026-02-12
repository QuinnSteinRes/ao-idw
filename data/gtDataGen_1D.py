import os
# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
import pandas as pd
import scipy.io

# Initial condition
def I(x):
    return np.sin(np.pi * x)

# Source term
def f(x, t):
    return 0

# Solve diffusion equation    
def solver_FE_simple(I, f, a, L, T, Nx):
    # von Neumann stability analysis
    dx = L / (Nx - 1)         # Constant mesh spacing in x
    dt = (0.5 / a) * dx**2    # Constant mesh spacing in t, F need to set up stability analysis 
    Nt = int(T / dt) + 1      # Mesh points in time or time instances, setting this to an integer (int) allows the total number of steps to be a whole number
    
    t = np.linspace(0, Nt * dt, Nt + 1)  # Mesh points in time
    x = np.linspace(0, L, Nx + 1)  # Mesh points in space

    # mesh Fourier number
    F = a * dt / dx ** 2
    
    # Make sure dx and dt are compatible with x and t
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # Make u 2D array
    u = np.zeros((Nt + 1, Nx + 1))  

    # Set initial condition u(x,0) = I(x)
    for i in range(0, Nx + 1):
        u[0, i] = I(x[i])
    # Insert boundary conditions
    u[:, 0] = 0
    u[:, Nx] = 0    

    for n in range(0, Nt):
        # Compute u at inner mesh points
        for i in range(1, Nx):
            u[n+1, i] = u[n,i] + F * (u[n,i-1] - 2*u[n,i] + u[n,i+1]) + dt * f(x[i], t[n])

    return u, x, t # Returning solutions

# Flow parameters
# Diffusion coefficient
a = 0.2
# Domain length
L = 1.0
# Duration
T = 1
# Number of meshes
Nx = 50

# Solution
u, x, t = solver_FE_simple(I, f, a, L, T, Nx)
print("nx =",x.shape[0],"nt =",t.shape[0], "u(nx,nt) =",u.shape)

# Save Pandas data
allData = {"usol": u, "x": x, "t": t}
scipy.io.savemat("/home/qpy/miniconda3/envs/nnmodelling/AO_new/gtDiff1D.mat", allData)

# Plotting
plt.figure(figsize=(12, 8))
plt.imshow(u.T, origin='lower', extent=[0, T, 0, L], aspect='auto', cmap='jet')
plt.colorbar(label='u')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Solution u(x, t)')
plt.show()
