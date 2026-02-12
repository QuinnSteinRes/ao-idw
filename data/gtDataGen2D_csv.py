"""
Step 1-4: Ground Truth 2D Diffusion Data Generator - Incremental CSV Testing

Physics IDENTICAL to gtDataGen2D.py:
- Dirichlet BCs (u=0 on all boundaries)
- IC: sin(π*x) * sin(π*y)
- D = 0.2

Steps:
- Step 1: Float coords [0,1], dense time
- Step 2: Integer pixel coords, dense time
- Step 3: Integer pixel coords, sparse time (8 frames like experimental)
- Step 4: Neumann BCs (experimental-like)
"""

import numpy as np
import pandas as pd
import scipy.io
import argparse
from pathlib import Path


def I(x, y):
    """Initial condition: sin(πx) * sin(πy) - identical to gtDataGen2D.py"""
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def solver_2D_FE_vectorized(a, Lx, Ly, T, Nx, Ny):
    """
    Vectorized 2D diffusion solver - identical to gtDataGen2D.py
    
    PDE: du/dt = a * (d²u/dx² + d²u/dy²)
    BCs: Dirichlet u=0 on all boundaries
    IC:  sin(πx) * sin(πy)
    """
    # Spatial discretization
    dx = Lx / Nx
    dy = Ly / Ny
    
    # Stability: dt <= 0.2 * min(dx², dy²) / a
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
    
    # Time stepping
    for n in range(Nt):
        # Interior points (vectorized)
        u[n+1, 1:-1, 1:-1] = (u[n, 1:-1, 1:-1] + 
                             Fx * (u[n, 0:-2, 1:-1] - 2*u[n, 1:-1, 1:-1] + u[n, 2:, 1:-1]) +
                             Fy * (u[n, 1:-1, 0:-2] - 2*u[n, 1:-1, 1:-1] + u[n, 1:-1, 2:]))
        # BCs: boundaries stay 0 (Dirichlet)
        
        if (n + 1) % 1000 == 0:
            print(f"  Time step {n+1}/{Nt}")
    
    return u, x, y, t


def solution_to_csv(u, x, y, t, use_integer_coords=True):
    """
    Convert solution array to CSV format with x, y, t, intensity columns.
    
    Args:
        use_integer_coords: If True, output x,y,t as integers (pixel/frame style)
    """
    nt, nx, ny = u.shape
    
    data = []
    for ti in range(nt):
        for xi in range(nx):
            for yi in range(ny):
                if use_integer_coords:
                    data.append({
                        'x': xi,
                        'y': yi,
                        't': ti,
                        'intensity': u[ti, xi, yi]
                    })
                else:
                    data.append({
                        'x': x[xi],
                        'y': y[yi],
                        't': t[ti],
                        'intensity': u[ti, xi, yi]
                    })
    
    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(
        description='Generate ground truth 2D diffusion data in CSV format (Step 1)'
    )
    parser.add_argument('--output', type=str, default='data/gtDiff2D_step1.csv',
                        help='Output CSV file path')
    parser.add_argument('--output-mat', type=str, default=None,
                        help='Also save .mat file for comparison (optional)')
    parser.add_argument('--diff-coeff', type=float, default=0.2,
                        help='Diffusion coefficient (default: 0.2, same as gtDataGen2D.py)')
    parser.add_argument('--nx', type=int, default=50,
                        help='Grid points in x (default: 50)')
    parser.add_argument('--ny', type=int, default=50,
                        help='Grid points in y (default: 50)')
    parser.add_argument('--t-subsample', type=int, default=10,
                        help='Subsample every N time steps (default: 10)')
    parser.add_argument('--step', type=int, default=2, choices=[1, 2, 3, 4],
                        help='Which step variant: 1=float coords, 2=int coords, 3=sparse time, 4=neumann')
    parser.add_argument('--n-frames', type=int, default=8,
                        help='Number of time frames for step 3+ (default: 8, like experimental)')
    
    args = parser.parse_args()
    
    # Physics parameters - IDENTICAL to gtDataGen2D.py defaults
    a = args.diff_coeff  # Diffusion coefficient
    Lx = 1.0             # Domain length x
    Ly = 1.0             # Domain length y
    T = 1.0              # Total time
    Nx = args.nx         # Grid points x
    Ny = args.ny         # Grid points y
    
    print("="*60)
    print(f"Step {args.step}: Ground Truth 2D Diffusion (CSV Output)")
    print("="*60)
    print(f"Physics IDENTICAL to gtDataGen2D.py:")
    print(f"  Diffusion coefficient: D = {a}")
    print(f"  Domain: [0, {Lx}] x [0, {Ly}]")
    print(f"  Time: [0, {T}]")
    print(f"  BCs: Dirichlet (u=0 on all boundaries)")
    print(f"  IC: sin(πx) * sin(πy)")
    print(f"  Coordinate format: {'integer (pixel-style)' if args.step >= 2 else 'float [0,1]'}")
    print(f"  Temporal sampling: {'sparse (' + str(args.n_frames) + ' frames)' if args.step >= 3 else 'dense'}")
    print()
    
    # Solve
    u, x, y, t = solver_2D_FE_vectorized(a, Lx, Ly, T, Nx, Ny)
    
    print(f"\nFull solution shape: {u.shape}")
    
    # Subsample time - Step 3+ uses sparse frames
    if args.step >= 3:
        # Sparse sampling: select n_frames evenly spaced
        n_frames = args.n_frames
        t_indices = np.linspace(0, len(t)-1, n_frames, dtype=int)
        print(f"Step 3+: Sparse temporal sampling ({n_frames} frames)")
    else:
        # Dense sampling
        t_indices = np.arange(0, len(t), args.t_subsample)
    
    u_sub = u[t_indices, :, :]
    t_sub = t[t_indices]
    
    print(f"Subsampled to {len(t_indices)} time steps")
    print(f"  u shape: {u_sub.shape}")
    print(f"  t values: {t_sub}")
    
    # Convert to CSV
    use_int_coords = args.step >= 2
    df = solution_to_csv(u_sub, x, y, t_sub, use_integer_coords=use_int_coords)
    
    print(f"\nCSV data:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  x range: [{df['x'].min():.4f}, {df['x'].max():.4f}]")
    print(f"  y range: [{df['y'].min():.4f}, {df['y'].max():.4f}]")
    print(f"  t range: [{df['t'].min():.4f}, {df['t'].max():.4f}]")
    print(f"  intensity range: [{df['intensity'].min():.4f}, {df['intensity'].max():.4f}]")
    
    # Save CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved CSV to: {output_path}")
    
    # Also save metadata for reference
    metadata = {
        'diffCoeff': a,
        'Lx': Lx,
        'Ly': Ly,
        'T': T,
        'Nx': Nx,
        'Ny': Ny,
        'bc_type': 'dirichlet',
        'ic_type': 'sin(pi*x)*sin(pi*y)',
        't_subsample': args.t_subsample,
        'n_time_steps': len(t_sub)
    }
    
    metadata_path = output_path.with_suffix('.metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write("Ground Truth 2D Diffusion - Step 1 Metadata\n")
        f.write("="*50 + "\n")
        for k, v in metadata.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved metadata to: {metadata_path}")
    
    # Optionally save .mat for direct comparison
    if args.output_mat:
        mat_data = {
            'usol': u_sub,
            'x': x,
            'y': y,
            't': t_sub,
            'diffCoeff': a
        }
        scipy.io.savemat(args.output_mat, mat_data)
        print(f"Saved .mat to: {args.output_mat}")
    
    print("\n" + "="*60)
    print("Step 1 Complete!")
    print("="*60)
    print("\nNext: Test this CSV with your PINN training script.")
    print("Expected result: Should converge to D ≈ 0.2 (same as .mat version)")
    
    return df, metadata


if __name__ == "__main__":
    main()