"""
Data loading utilities for 2D diffusion inverse problems.
Extracted from legacy: trainingdata_2D() function
"""
import numpy as np
import scipy.io
from pyDOE import lhs


def load_2d_diffusion_data(config):
    """
    Load 2D diffusion ground truth data and create training/test sets.
    
    Args:
        config: Config object with data parameters
        
    Returns:
        dict: Contains training data, test data, bounds, and ground truth
            - X_f_train: Collocation points for PDE residual (n_f, 3)
            - X_u_train: BC/IC points (n_u, 3)
            - u_train: BC/IC values (n_u, 1)
            - X_obs: Interior observation points for inverse problem (n_obs, 3)
            - u_obs: Interior observation values (n_obs, 1)
            - X_u_test: All space-time test points (N_total, 3)
            - u_test: All solution values (N_total, 1)
            - bounds: (lb, ub) domain bounds
            - grid: (x, y, t) coordinate arrays
            - usol: Reshaped solution (Nx+1, Ny+1, Nt)
            - diff_coeff_true: True diffusion coefficient from file
    """
    # Load ground truth data
    data = scipy.io.loadmat(config.data.input_file)
    x = data['x'].flatten()           # (Nx+1,)
    y = data['y'].flatten()           # (Ny+1,)
    t = data['t'].flatten()           # (Nt,)
    usol = data['usol']               # (Nt, Nx+1, Ny+1)
    
    # Get true diffusion coefficient if stored
    diff_coeff_true = float(data.get('diffCoeff', config.physics.diff_coeff_true))
    
    # Transpose to (Nx+1, Ny+1, Nt) indexing
    usol_reshaped = np.transpose(usol, (1, 2, 0))
    
    # Create meshgrid for all space-time points
    X, Y, T = np.meshgrid(x, y, t, indexing='ij')
    
    # Test data: all space-time points
    X_u_test = np.hstack((X.flatten()[:, None], 
                          Y.flatten()[:, None], 
                          T.flatten()[:, None]))
    u_test = usol_reshaped.flatten()[:, None]
    
    # Domain bounds
    lb = np.array([x.min(), y.min(), t.min()])
    ub = np.array([x.max(), y.max(), t.max()])
    
    # Create BC/IC training sets
    X_u_train, u_train = _create_bc_ic_points(x, y, t, usol_reshaped, config.data.n_u)
    
    # Create collocation points via LHS
    X_f_train = _create_collocation_points(lb, ub, config.data.n_f, X_u_train)
    
    # Create interior observation points for inverse problem
    X_obs, u_obs = _create_observation_points(X_u_test, u_test, lb, ub, config.data.n_obs)
    
    return {
        'X_f_train': X_f_train,
        'X_u_train': X_u_train,
        'u_train': u_train,
        'X_obs': X_obs,
        'u_obs': u_obs,
        'X_u_test': X_u_test,
        'u_test': u_test,
        'bounds': (lb, ub),
        'grid': (x, y, t),
        'usol': usol_reshaped,
        'diff_coeff_true': diff_coeff_true
    }


def _create_bc_ic_points(x, y, t, usol, n_u):
    """
    Create boundary and initial condition training points.
    
    Boundaries: x_min, x_max, y_min, y_max for all (y,t) or (x,t)
    Initial: t=0 for all (x,y)
    
    Returns:
        X_u_train: Sampled BC/IC points (n_u, 3)
        u_train: Corresponding values (n_u, 1)
    """
    all_X_u = []
    all_u = []
    
    # Initial Condition: t=0, all x, all y
    X_ic, Y_ic = np.meshgrid(x, y, indexing='ij')
    T_ic = np.zeros_like(X_ic)
    ic_points = np.hstack((X_ic.flatten()[:, None], 
                           Y_ic.flatten()[:, None], 
                           T_ic.flatten()[:, None]))
    ic_values = usol[:, :, 0].flatten()[:, None]
    all_X_u.append(ic_points)
    all_u.append(ic_values)
    
    # BC at x = x_min (for all y, all t)
    Y_bc, T_bc = np.meshgrid(y, t, indexing='ij')
    X_bc = np.ones_like(Y_bc) * x.min()
    bc_x_min = np.hstack((X_bc.flatten()[:, None], 
                          Y_bc.flatten()[:, None], 
                          T_bc.flatten()[:, None]))
    bc_x_min_u = usol[0, :, :].flatten()[:, None]
    all_X_u.append(bc_x_min)
    all_u.append(bc_x_min_u)
    
    # BC at x = x_max
    X_bc = np.ones_like(Y_bc) * x.max()
    bc_x_max = np.hstack((X_bc.flatten()[:, None], 
                          Y_bc.flatten()[:, None], 
                          T_bc.flatten()[:, None]))
    bc_x_max_u = usol[-1, :, :].flatten()[:, None]
    all_X_u.append(bc_x_max)
    all_u.append(bc_x_max_u)
    
    # BC at y = y_min (for all x, all t)
    X_bc, T_bc = np.meshgrid(x, t, indexing='ij')
    Y_bc = np.ones_like(X_bc) * y.min()
    bc_y_min = np.hstack((X_bc.flatten()[:, None], 
                          Y_bc.flatten()[:, None], 
                          T_bc.flatten()[:, None]))
    bc_y_min_u = usol[:, 0, :].flatten()[:, None]
    all_X_u.append(bc_y_min)
    all_u.append(bc_y_min_u)
    
    # BC at y = y_max
    Y_bc = np.ones_like(X_bc) * y.max()
    bc_y_max = np.hstack((X_bc.flatten()[:, None], 
                          Y_bc.flatten()[:, None], 
                          T_bc.flatten()[:, None]))
    bc_y_max_u = usol[:, -1, :].flatten()[:, None]
    all_X_u.append(bc_y_max)
    all_u.append(bc_y_max_u)
    
    # Stack all BC/IC points
    all_X_u = np.vstack(all_X_u)
    all_u = np.vstack(all_u)
    
    # Randomly sample n_u points
    idx = np.random.choice(all_X_u.shape[0], min(n_u, all_X_u.shape[0]), replace=False)
    return all_X_u[idx, :], all_u[idx, :]


def _create_collocation_points(lb, ub, n_f, X_u_train):
    """
    Create collocation points for PDE residual via Latin Hypercube Sampling.
    Appends BC/IC points to ensure coverage.
    
    Returns:
        X_f_train: Collocation points (n_f + n_u, 3)
    """
    # LHS in normalized [0,1]^3 space, then scale to domain
    X_f = lb + (ub - lb) * lhs(3, n_f)  # 3D: (x, y, t)
    
    # Append BC/IC points for better coverage
    X_f_train = np.vstack((X_f, X_u_train))
    
    return X_f_train


def _create_observation_points(X_u_test, u_test, lb, ub, n_obs):
    """
    Create interior observation points for inverse problem.
    Excludes 5% margin from boundaries to avoid BC overlap.
    
    Returns:
        X_obs: Interior observation points (n_obs, 3)
        u_obs: Corresponding values (n_obs, 1)
    """
    # 5% margin from boundaries
    eps = 0.05 * (ub - lb)
    
    # Mask for interior points (excluding boundaries and initial time)
    interior_mask = (
        (X_u_test[:, 0] > lb[0] + eps[0]) & (X_u_test[:, 0] < ub[0] - eps[0]) &
        (X_u_test[:, 1] > lb[1] + eps[1]) & (X_u_test[:, 1] < ub[1] - eps[1]) &
        (X_u_test[:, 2] > lb[2] + eps[2]) & (X_u_test[:, 2] < ub[2] - eps[2])
    )
    
    X_interior = X_u_test[interior_mask]
    u_interior = u_test[interior_mask]
    
    # Randomly sample n_obs observation points
    idx = np.random.choice(X_interior.shape[0], 
                          min(n_obs, X_interior.shape[0]), 
                          replace=False)
    
    return X_interior[idx, :], u_interior[idx, :]