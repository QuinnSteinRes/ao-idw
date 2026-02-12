import os
# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# Initial condition for 2D diffusion
def I(x, y):
    """Initial condition: product of sine functions in x and y"""
    return np.sin(np.pi * x) * np.sin(np.pi * y)

# Source term (zero for pure diffusion)
def f(x, y, t):
    return 0

def solver_2D_FE(I, f, a, Lx, Ly, T, Nx, Ny):
    """
    Solve 2D diffusion equation using Forward Euler (explicit) method.
    
    PDE: du/dt = a * (d²u/dx² + d²u/dy²) + f(x,y,t)
    
    Parameters:
    -----------
    I : function
        Initial condition I(x, y)
    f : function  
        Source term f(x, y, t)
    a : float
        Diffusion coefficient
    Lx, Ly : float
        Domain lengths in x and y directions
    T : float
        Total simulation time
    Nx, Ny : int
        Number of spatial grid points in x and y
        
    Returns:
    --------
    u : ndarray
        Solution array of shape (Nt+1, Nx+1, Ny+1)
    x : ndarray
        x coordinate array
    y : ndarray
        y coordinate array
    t : ndarray
        time coordinate array
    """
    # Spatial discretization
    dx = Lx / (Nx)
    dy = Ly / (Ny)
    
    # Stability condition for 2D explicit scheme: dt <= dx^2 * dy^2 / (2*a*(dx^2 + dy^2))
    # Using a more conservative estimate
    dt = 0.25 * min(dx**2, dy**2) / a
    
    Nt = int(T / dt) + 1
    
    # Create mesh
    x = np.linspace(0, Lx, Nx + 1)
    y = np.linspace(0, Ly, Ny + 1)
    t = np.linspace(0, Nt * dt, Nt + 1)
    
    # Recalculate dt to match grid
    dt = t[1] - t[0]
    
    # Mesh Fourier numbers
    Fx = a * dt / dx**2
    Fy = a * dt / dy**2
    
    print(f"Grid: Nx={Nx+1}, Ny={Ny+1}, Nt={Nt+1}")
    print(f"dx={dx:.6f}, dy={dy:.6f}, dt={dt:.6f}")
    print(f"Fourier numbers: Fx={Fx:.4f}, Fy={Fy:.4f}")
    print(f"Stability check (Fx + Fy < 0.5): {Fx + Fy:.4f} < 0.5 = {Fx + Fy < 0.5}")
    
    # Initialize solution array
    u = np.zeros((Nt + 1, Nx + 1, Ny + 1))
    
    # Set initial condition
    X, Y = np.meshgrid(x, y, indexing='ij')
    u[0, :, :] = I(X, Y)
    
    # Apply boundary conditions (Dirichlet: u = 0 on all boundaries)
    u[:, 0, :] = 0   # x = 0
    u[:, -1, :] = 0  # x = Lx
    u[:, :, 0] = 0   # y = 0
    u[:, :, -1] = 0  # y = Ly
    
    # Time stepping using Forward Euler
    for n in range(Nt):
        for i in range(1, Nx):
            for j in range(1, Ny):
                u[n+1, i, j] = (u[n, i, j] + 
                               Fx * (u[n, i-1, j] - 2*u[n, i, j] + u[n, i+1, j]) +
                               Fy * (u[n, i, j-1] - 2*u[n, i, j] + u[n, i, j+1]) +
                               dt * f(x[i], y[j], t[n]))
        
        # Progress indicator for long simulations
        if (n + 1) % 1000 == 0:
            print(f"  Time step {n+1}/{Nt}")
    
    return u, x, y, t

def solver_2D_FE_vectorized(I, f, a, Lx, Ly, T, Nx, Ny):
    """
    Vectorized version of 2D diffusion solver for better performance.
    """
    # Spatial discretization
    dx = Lx / Nx
    dy = Ly / Ny
    
    # Stability condition for 2D: Fx + Fy < 0.5
    # Use 0.2 factor for safety margin
    dt = 0.2 * min(dx**2, dy**2) / a
    
    Nt = int(T / dt) + 1
    
    # Create mesh
    x = np.linspace(0, Lx, Nx + 1)
    y = np.linspace(0, Ly, Ny + 1)
    t = np.linspace(0, Nt * dt, Nt + 1)
    
    dt = t[1] - t[0]
    
    # Mesh Fourier numbers
    Fx = a * dt / dx**2
    Fy = a * dt / dy**2
    
    print(f"Grid: Nx={Nx+1}, Ny={Ny+1}, Nt={Nt+1}")
    print(f"dx={dx:.6f}, dy={dy:.6f}, dt={dt:.6f}")
    print(f"Fourier numbers: Fx={Fx:.4f}, Fy={Fy:.4f}")
    print(f"Stability check (Fx + Fy < 0.5): {Fx + Fy:.4f} < 0.5 = {Fx + Fy < 0.5}")
    
    # Initialize solution array
    u = np.zeros((Nt + 1, Nx + 1, Ny + 1))
    
    # Set initial condition
    X, Y = np.meshgrid(x, y, indexing='ij')
    u[0, :, :] = I(X, Y)
    
    # Time stepping using vectorized operations
    for n in range(Nt):
        # Interior points update (vectorized)
        u[n+1, 1:-1, 1:-1] = (u[n, 1:-1, 1:-1] + 
                             Fx * (u[n, 0:-2, 1:-1] - 2*u[n, 1:-1, 1:-1] + u[n, 2:, 1:-1]) +
                             Fy * (u[n, 1:-1, 0:-2] - 2*u[n, 1:-1, 1:-1] + u[n, 1:-1, 2:]))
        
        # Boundary conditions are already 0 from initialization
        
        if (n + 1) % 1000 == 0:
            print(f"  Time step {n+1}/{Nt}")
    
    return u, x, y, t

# ============================================================
# Main execution
# ============================================================
if __name__ == "__main__":
    # Physical parameters
    a = 0.2        # Diffusion coefficient
    Lx = 1.0       # Domain length in x
    Ly = 1.0       # Domain length in y
    T = 1.0        # Total simulation time
    
    # Grid parameters
    Nx = 50        # Number of grid points in x
    Ny = 50        # Number of grid points in y
    
    print("Solving 2D diffusion equation...")
    print(f"Diffusion coefficient: a = {a}")
    print(f"Domain: [0, {Lx}] x [0, {Ly}]")
    print(f"Time: [0, {T}]")
    print()
    
    # Solve using vectorized method (faster)
    u, x, y, t = solver_2D_FE_vectorized(I, f, a, Lx, Ly, T, Nx, Ny)
    
    print(f"\nSolution shape: u(x, y, t) has shape {u.shape}")
    print(f"  x: {x.shape}")
    print(f"  y: {y.shape}")  
    print(f"  t: {t.shape}")
    
    # Subsample time for manageable file size
    # Save every nth time step
    t_subsample = 10
    t_indices = np.arange(0, len(t), t_subsample)
    u_subsampled = u[t_indices, :, :]
    t_subsampled = t[t_indices]
    
    print(f"\nSubsampled to {len(t_indices)} time steps")
    print(f"  u shape: {u_subsampled.shape}")
    print(f"  t shape: {t_subsampled.shape}")
    
    # Save data
    output_file = "gtDiff2D.mat"
    allData = {
        "usol": u_subsampled,  # Shape: (Nt_sub, Nx+1, Ny+1)
        "x": x,                 # Shape: (Nx+1,)
        "y": y,                 # Shape: (Ny+1,)
        "t": t_subsampled,      # Shape: (Nt_sub,)
        "diffCoeff": a
    }
    scipy.io.savemat(output_file, allData)
    print(f"\nSaved to {output_file}")
    
    # ============================================================
    # Plotting
    # ============================================================
    
    # Create meshgrid for plotting
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Figure 1: Solution at different times
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    # Select time indices to plot
    plot_times = [0, len(t_indices)//5, 2*len(t_indices)//5, 
                  3*len(t_indices)//5, 4*len(t_indices)//5, -1]
    
    for idx, (ax, ti) in enumerate(zip(axes.flatten(), plot_times)):
        im = ax.pcolormesh(X, Y, u_subsampled[ti, :, :], 
                          cmap='jet', shading='auto', vmin=0, vmax=1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f't = {t_subsampled[ti]:.3f}s')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='u')
    
    plt.suptitle(f'2D Diffusion: u(x,y,t), D = {a}', fontsize=14)
    plt.tight_layout()
    plt.savefig('diff2D_evolution.png', dpi=150)
    print("Saved diff2D_evolution.png")
    
    # Figure 2: Cross-sections
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Cross-section at y = Ly/2
    mid_y = Ny // 2
    for ti in [0, len(t_indices)//4, len(t_indices)//2, 3*len(t_indices)//4, -1]:
        axes[0].plot(x, u_subsampled[ti, :, mid_y], 
                    label=f't = {t_subsampled[ti]:.2f}s')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u(x, y=0.5, t)')
    axes[0].set_title('Cross-section at y = 0.5')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Cross-section at x = Lx/2
    mid_x = Nx // 2
    for ti in [0, len(t_indices)//4, len(t_indices)//2, 3*len(t_indices)//4, -1]:
        axes[1].plot(y, u_subsampled[ti, mid_x, :], 
                    label=f't = {t_subsampled[ti]:.2f}s')
    axes[1].set_xlabel('y')
    axes[1].set_ylabel('u(x=0.5, y, t)')
    axes[1].set_title('Cross-section at x = 0.5')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Maximum value over time
    u_max = np.max(u_subsampled, axis=(1, 2))
    axes[2].plot(t_subsampled, u_max, 'b-', linewidth=2)
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('max(u)')
    axes[2].set_title('Maximum value decay')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('diff2D_crosssections.png', dpi=150)
    print("Saved diff2D_crosssections.png")
    
    # Figure 3: 3D surface plot at specific time
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, u_subsampled[0, :, :], 
                            cmap='jet', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u')
    ax1.set_title('Initial condition t = 0')
    
    ax2 = fig.add_subplot(122, projection='3d')
    mid_t = len(t_indices) // 2
    surf2 = ax2.plot_surface(X, Y, u_subsampled[mid_t, :, :], 
                            cmap='jet', alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u')
    ax2.set_title(f't = {t_subsampled[mid_t]:.3f}s')
    
    plt.tight_layout()
    plt.savefig('diff2D_3Dsurface.png', dpi=150)
    print("Saved diff2D_3Dsurface.png")
    
    plt.show()
    
    print("\nDone!")