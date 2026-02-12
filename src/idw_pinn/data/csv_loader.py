"""
CSV data loading utilities for 2D diffusion inverse problems.

Handles both synthetic and experimental CSV data in the format:
    x, y, t, intensity

This loader converts pixel/frame units to normalized physical units,
extracts boundary conditions from ROI edges, and prepares data for PINN training.

Key differences from MAT loader:
1. CSV data is typically in pixel/frame units (integers)
2. Boundary conditions are extracted from ROI edges (not true physical BCs)
3. Initial condition is derived from t=0 data
4. May need interpolation for irregular experimental sampling
"""
import numpy as np
import pandas as pd

def _stratified_split_by_time(X: np.ndarray, u: np.ndarray, val_frac: float = 0.10, seed: int = 1234):
    """Stratified train/val split by unique time values in X[:,2]."""
    if X is None or len(X) == 0:
        return X, u, np.zeros((0, 3), dtype=float), np.zeros((0, 1), dtype=float)

    rng = np.random.default_rng(seed)
    tvals = X[:, 2]
    uniq = np.unique(tvals)

    train_idx = []
    val_idx = []
    for t in uniq:
        idx = np.where(tvals == t)[0]
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        n_val = max(1, int(round(val_frac * len(idx))))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])

    train_idx = np.concatenate(train_idx) if len(train_idx) else np.array([], dtype=int)
    val_idx = np.concatenate(val_idx) if len(val_idx) else np.array([], dtype=int)

    return X[train_idx], u[train_idx], X[val_idx], u[val_idx]

from pathlib import Path
from pyDOE import lhs
from scipy.interpolate import griddata


def load_pointcloud_csv_data(config, df=None, csv_path=None):
    """Load a point-cloud CSV exported from the masking pipeline.

    Expected columns: x, y, t, u

    - x, y are already normalized to [0,1] in the dish bounding box
    - t is in seconds (or whatever physical time unit you choose)
    - u is normalized intensity in [0,1]

    This loader intentionally does **not** invent rectangular boundary conditions.
    It returns empty BC/IC arrays by default and uses the CSV points as the
    interior observation set.
    """
    if df is None:
        csv_path = Path(config.data.input_file) if csv_path is None else csv_path
        df = pd.read_csv(csv_path)
    if csv_path is None:
        csv_path = Path(getattr(config.data, 'input_file', 'dataset.csv'))

    required_cols = ['x', 'y', 't']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    # Observation column can be either 'u' or 'intensity'
    obs_col = 'u' if 'u' in df.columns else 'intensity' if 'intensity' in df.columns else None
    if obs_col is None:
        raise ValueError("Pointcloud CSV must contain column 'u' (preferred) or 'intensity'")

    X_all = df[['x', 'y', 't']].to_numpy(dtype=float)
    u_all = df[[obs_col]].to_numpy(dtype=float)

    # Optional subsampling for very large point clouds
    n_obs = getattr(config.data, 'n_obs', None)
    max_obs = getattr(config.data, 'max_obs', None)
    target = None
    if isinstance(n_obs, (int, np.integer)) and n_obs > 0:
        target = int(n_obs)
    if isinstance(max_obs, (int, np.integer)) and max_obs > 0:
        target = int(max_obs) if target is None else min(target, int(max_obs))
    if target is not None and target < len(X_all):
        rng = np.random.default_rng(getattr(config.training, 'seed', 1234) if hasattr(config, 'training') else 1234)
        idx = rng.choice(len(X_all), size=target, replace=False)
        X_obs = X_all[idx]
        u_obs = u_all[idx]
    else:
        X_obs, u_obs = X_all, u_all

    # Held-out validation split (10%) stratified by time t
    seed = int(getattr(config.training, 'seed', 1234) if hasattr(config, 'training') else 1234)
    X_obs, u_obs, X_obs_val, u_obs_val = _stratified_split_by_time(X_obs, u_obs, val_frac=0.10, seed=seed)

    # Domain bounds from data (already normalized coords)
    lb = X_all.min(axis=0)
    ub = X_all.max(axis=0)

    # Collocation points via LHS; also include observation points to help enforce PDE where data exists
    n_f = int(getattr(config.data, 'n_f', 20000))
    X_f_train = _create_collocation_points(lb, ub, n_f, X_obs)

    # No BCs by default (unless user wants to add them explicitly)
    X_u_train = np.zeros((0, 3), dtype=float)
    u_train = np.zeros((0, 1), dtype=float)

    metadata = {
        'csv_mode': 'pointcloud',
        'obs_col': obs_col,
        'n_total': int(len(df)),
        'n_obs_used': int(len(X_obs)),
        'n_obs_val': int(len(X_obs_val)),
        'bounds': {'lb': lb.tolist(), 'ub': ub.tolist()},
    }

    diff_coeff_true = _load_ground_truth(Path(csv_path), config)

    return {
        'X_f_train': X_f_train,
        'X_u_train': X_u_train,
        'u_train': u_train,
        'X_obs': X_obs,
        'u_obs': u_obs,
        'X_u_test': X_all,
        'u_test': u_all,
        'bounds': (lb, ub),
        'grid': None,
        'usol': None,
        'diff_coeff_true': diff_coeff_true,
        'metadata': metadata,
    }


def load_csv_diffusion_data(config):
    """
    Load CSV diffusion data and create training/test sets.
    
    Handles the CSV format from experimental microscopy data:
    - x, y: pixel coordinates
    - t: frame numbers
    - intensity: grayscale values (proportional to concentration)
    
    Args:
        config: Config object with data parameters including:
            - data.input_file: path to CSV file
            - data.n_u: number of BC/IC points
            - data.n_f: number of collocation points
            - data.n_obs: number of interior observation points
            - data.normalize_coords: whether to normalize to [0,1] (default True)
            - data.normalize_intensity: whether to normalize intensity (default True)
            
    Returns:
        dict: Contains training data, test data, bounds, and metadata
            - X_f_train: Collocation points for PDE residual
            - X_u_train: BC/IC points
            - u_train: BC/IC values
            - X_obs: Interior observation points
            - u_obs: Interior observation values
            - X_u_test: All space-time test points
            - u_test: All solution values
            - bounds: (lb, ub) domain bounds
            - grid: (x_unique, y_unique, t_unique) coordinate arrays
            - usol: Reshaped solution (nx, ny, nt)
            - diff_coeff_true: True diffusion coefficient (if available)
            - metadata: Dict with normalization factors and original ranges
    """
    # Load CSV
    csv_path = Path(config.data.input_file)
    df = pd.read_csv(csv_path)

    # ------------------------------------------------------------
    # Fast path: point-cloud CSV produced by our masking/extraction script
    # Columns: x, y, t, u (already in normalized coordinates and intensity)
    # ------------------------------------------------------------
    csv_mode = getattr(config.data, 'csv_mode', None)
    if csv_mode == 'pointcloud' or ('u' in df.columns and 'intensity' not in df.columns):
        return load_pointcloud_csv_data(config, df, csv_path)
    
    # Validate columns (grid-style CSV)
    required_cols = ['x', 'y', 't', 'intensity']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Extract unique coordinates
    x_unique = np.sort(df['x'].unique())
    y_unique = np.sort(df['y'].unique())
    t_unique = np.sort(df['t'].unique())
    
    nx = len(x_unique)
    ny = len(y_unique)
    nt = len(t_unique)
    
    print(f"CSV data loaded: {len(df)} points")
    print(f"Grid: nx={nx}, ny={ny}, nt={nt}")
    print(f"X range: [{x_unique.min()}, {x_unique.max()}]")
    print(f"Y range: [{y_unique.min()}, {y_unique.max()}]")
    print(f"T range: [{t_unique.min()}, {t_unique.max()}]")
    
    # Check data completeness
    expected_points = nx * ny * nt
    if len(df) != expected_points:
        print(f"Warning: Expected {expected_points} points, got {len(df)}")
        print("Data may have missing values - will interpolate")
    
    # Store original ranges for metadata
    metadata = {
        'x_range': (x_unique.min(), x_unique.max()),
        'y_range': (y_unique.min(), y_unique.max()),
        't_range': (t_unique.min(), t_unique.max()),
        'intensity_range': (df['intensity'].min(), df['intensity'].max()),
        'nx': nx,
        'ny': ny,
        'nt': nt
    }
    
    # Normalize coordinates to [0, 1] if requested
    normalize_coords = getattr(config.data, 'normalize_coords', True)
    normalize_intensity = getattr(config.data, 'normalize_intensity', True)
    
    if normalize_coords:
        # Create normalized coordinate arrays
        x_norm = (x_unique - x_unique.min()) / (x_unique.max() - x_unique.min())
        y_norm = (y_unique - y_unique.min()) / (y_unique.max() - y_unique.min())
        t_norm = (t_unique - t_unique.min()) / (t_unique.max() - t_unique.min())
        
        # Store normalization factors
        metadata['x_scale'] = x_unique.max() - x_unique.min()
        metadata['y_scale'] = y_unique.max() - y_unique.min()
        metadata['t_scale'] = t_unique.max() - t_unique.min()
        metadata['x_offset'] = x_unique.min()
        metadata['y_offset'] = y_unique.min()
        metadata['t_offset'] = t_unique.min()
    else:
        x_norm = x_unique.astype(float)
        y_norm = y_unique.astype(float)
        t_norm = t_unique.astype(float)
    
    # Reshape intensity to 3D array
    usol = _reshape_csv_to_array(df, x_unique, y_unique, t_unique, normalize_coords)
    
    # Normalize intensity if requested
    if normalize_intensity:
        i_min, i_max = usol.min(), usol.max()
        if i_max > i_min:
            usol = (usol - i_min) / (i_max - i_min)
        metadata['intensity_scale'] = i_max - i_min
        metadata['intensity_offset'] = i_min
    
    print(f"Solution array shape: {usol.shape}")
    print(f"Intensity range (normalized): [{usol.min():.3f}, {usol.max():.3f}]")
    
    # Create meshgrid for all space-time points
    X, Y, T = np.meshgrid(x_norm, y_norm, t_norm, indexing='ij')
    
    # Test data: all space-time points
    X_u_test = np.hstack((X.flatten()[:, None], 
                          Y.flatten()[:, None], 
                          T.flatten()[:, None]))
    u_test = usol.flatten()[:, None]
    
    # Domain bounds (after normalization)
    lb = np.array([x_norm.min(), y_norm.min(), t_norm.min()])
    ub = np.array([x_norm.max(), y_norm.max(), t_norm.max()])
    
    # Create BC/IC training sets
    X_u_train, u_train = _create_bc_ic_from_csv(
        x_norm, y_norm, t_norm, usol, config.data.n_u
    )
    
    # Create collocation points via LHS
    X_f_train = _create_collocation_points(lb, ub, config.data.n_f, X_u_train)
    
    # Create interior observation points
    X_obs, u_obs = _create_observation_points_csv(
        X_u_test, u_test, lb, ub, config.data.n_obs
    )
    
    # Try to load ground truth diffusion coefficient if available
    diff_coeff_true = _load_ground_truth(csv_path, config)
    
    return {
        'X_f_train': X_f_train,
        'X_u_train': X_u_train,
        'u_train': u_train,
        'X_obs': X_obs,
        'u_obs': u_obs,
        'X_u_test': X_u_test,
        'u_test': u_test,
        'bounds': (lb, ub),
        'grid': (x_norm, y_norm, t_norm),
        'usol': usol,
        'diff_coeff_true': diff_coeff_true,
        'metadata': metadata
    }


def _reshape_csv_to_array(df, x_unique, y_unique, t_unique, normalize_coords):
    """
    Reshape CSV data to 3D array (nx, ny, nt).
    
    Handles potentially missing data points through interpolation.
    """
    nx = len(x_unique)
    ny = len(y_unique)
    nt = len(t_unique)
    
    # Create mapping from coordinates to indices
    x_to_idx = {x: i for i, x in enumerate(x_unique)}
    y_to_idx = {y: i for i, y in enumerate(y_unique)}
    t_to_idx = {t: i for i, t in enumerate(t_unique)}
    
    # Initialize array
    usol = np.full((nx, ny, nt), np.nan)
    
    # Fill array from DataFrame
    for _, row in df.iterrows():
        xi = x_to_idx.get(row['x'])
        yi = y_to_idx.get(row['y'])
        ti = t_to_idx.get(row['t'])
        if xi is not None and yi is not None and ti is not None:
            usol[xi, yi, ti] = row['intensity']
    
    # Check for and handle NaN values
    nan_count = np.isnan(usol).sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} missing values detected, interpolating...")
        usol = _interpolate_missing(usol, x_unique, y_unique, t_unique)
    
    return usol


def _interpolate_missing(usol, x, y, t):
    """
    Interpolate missing values in solution array.
    
    Uses nearest-neighbor interpolation for robustness with sparse data.
    """
    nx, ny, nt = usol.shape
    X, Y, T = np.meshgrid(x, y, t, indexing='ij')
    
    # Get known points
    known_mask = ~np.isnan(usol)
    known_points = np.column_stack([
        X[known_mask],
        Y[known_mask],
        T[known_mask]
    ])
    known_values = usol[known_mask]
    
    # Get unknown points
    unknown_mask = np.isnan(usol)
    unknown_points = np.column_stack([
        X[unknown_mask],
        Y[unknown_mask],
        T[unknown_mask]
    ])
    
    # Interpolate
    interpolated = griddata(
        known_points, known_values, unknown_points,
        method='nearest'
    )
    
    usol[unknown_mask] = interpolated
    return usol


def _create_bc_ic_from_csv(x, y, t, usol, n_u):
    """
    Create boundary and initial condition training points from CSV data.
    
    For experimental ROI data, "boundary conditions" are the values at
    the edges of the ROI (not true physical boundaries). This is handled
    with Neumann-like soft constraints.
    
    Returns:
        X_u_train: Sampled BC/IC points (n_u, 3)
        u_train: Corresponding values (n_u, 1)
    """
    nx, ny, nt = usol.shape
    all_X_u = []
    all_u = []
    
    # Initial Condition: t = t_min, all x, all y
    X_ic, Y_ic = np.meshgrid(x, y, indexing='ij')
    T_ic = np.full_like(X_ic, t.min())
    ic_points = np.hstack((X_ic.flatten()[:, None], 
                           Y_ic.flatten()[:, None], 
                           T_ic.flatten()[:, None]))
    ic_values = usol[:, :, 0].flatten()[:, None]
    all_X_u.append(ic_points)
    all_u.append(ic_values)
    
    # Boundary conditions at edges of ROI (for all t > 0)
    # These are "soft" BCs - we're matching the ROI edge values
    
    # x = x_min edge
    Y_bc, T_bc = np.meshgrid(y, t[1:], indexing='ij')  # Skip t=0 (in IC)
    X_bc = np.full_like(Y_bc, x.min())
    bc_x_min = np.hstack((X_bc.flatten()[:, None], 
                          Y_bc.flatten()[:, None], 
                          T_bc.flatten()[:, None]))
    bc_x_min_u = usol[0, :, 1:].flatten()[:, None]
    all_X_u.append(bc_x_min)
    all_u.append(bc_x_min_u)
    
    # x = x_max edge
    X_bc = np.full_like(Y_bc, x.max())
    bc_x_max = np.hstack((X_bc.flatten()[:, None], 
                          Y_bc.flatten()[:, None], 
                          T_bc.flatten()[:, None]))
    bc_x_max_u = usol[-1, :, 1:].flatten()[:, None]
    all_X_u.append(bc_x_max)
    all_u.append(bc_x_max_u)
    
    # y = y_min edge
    X_bc, T_bc = np.meshgrid(x, t[1:], indexing='ij')
    Y_bc = np.full_like(X_bc, y.min())
    bc_y_min = np.hstack((X_bc.flatten()[:, None], 
                          Y_bc.flatten()[:, None], 
                          T_bc.flatten()[:, None]))
    bc_y_min_u = usol[:, 0, 1:].flatten()[:, None]
    all_X_u.append(bc_y_min)
    all_u.append(bc_y_min_u)
    
    # y = y_max edge
    Y_bc = np.full_like(X_bc, y.max())
    bc_y_max = np.hstack((X_bc.flatten()[:, None], 
                          Y_bc.flatten()[:, None], 
                          T_bc.flatten()[:, None]))
    bc_y_max_u = usol[:, -1, 1:].flatten()[:, None]
    all_X_u.append(bc_y_max)
    all_u.append(bc_y_max_u)
    
    # Stack all BC/IC points
    all_X_u = np.vstack(all_X_u)
    all_u = np.vstack(all_u)
    
    print(f"Total BC/IC points available: {len(all_X_u)}")
    print(f"  - Initial condition: {nx * ny}")
    print(f"  - Boundary (x edges): {2 * ny * (nt-1)}")
    print(f"  - Boundary (y edges): {2 * nx * (nt-1)}")
    
    # Randomly sample n_u points
    n_sample = min(n_u, len(all_X_u))
    idx = np.random.choice(len(all_X_u), n_sample, replace=False)
    
    return all_X_u[idx, :], all_u[idx, :]


def _create_collocation_points(lb, ub, n_f, X_u_train):
    """
    Create collocation points for PDE residual via Latin Hypercube Sampling.
    """
    X_f = lb + (ub - lb) * lhs(3, n_f)
    X_f_train = np.vstack((X_f, X_u_train))
    return X_f_train


def _create_observation_points_csv(X_u_test, u_test, lb, ub, n_obs):
    """
    Create interior observation points for inverse problem.
    
    Uses smaller margins (2%) than MAT loader since CSV data is typically
    from a smaller ROI where all points are potentially useful.
    """
    # 2% margin from boundaries
    eps = 0.02 * (ub - lb)
    
    # Mask for interior points
    interior_mask = (
        (X_u_test[:, 0] > lb[0] + eps[0]) & (X_u_test[:, 0] < ub[0] - eps[0]) &
        (X_u_test[:, 1] > lb[1] + eps[1]) & (X_u_test[:, 1] < ub[1] - eps[1]) &
        (X_u_test[:, 2] > lb[2] + eps[2]) & (X_u_test[:, 2] < ub[2] - eps[2])
    )
    
    X_interior = X_u_test[interior_mask]
    u_interior = u_test[interior_mask]
    
    print(f"Interior points available: {len(X_interior)}")
    
    # Randomly sample n_obs observation points
    n_sample = min(n_obs, len(X_interior))
    idx = np.random.choice(len(X_interior), n_sample, replace=False)
    
    return X_interior[idx, :], u_interior[idx, :]


def _load_ground_truth(csv_path, config):
    """
    Try to load ground truth diffusion coefficient from companion files.
    
    Checks for:
    1. JSON metadata file (from synthetic data generator)
    2. Config file setting
    """
    # Try JSON metadata
    json_path = csv_path.with_suffix('.json')
    if json_path.exists():
        import json
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        if 'diff_coeff_true' in metadata:
            print(f"Loaded ground truth D = {metadata['diff_coeff_true']} from {json_path}")
            return metadata['diff_coeff_true']
    
    # Try config
    if hasattr(config, 'physics') and hasattr(config.physics, 'diff_coeff_true'):
        return config.physics.diff_coeff_true
    
    # No ground truth available
    print("No ground truth diffusion coefficient available")
    return None


def convert_diffusion_coefficient(D_normalized, metadata, pixel_size_um=None, frame_time_s=None):
    """
    Convert learned diffusion coefficient from normalized units to physical units.
    
    Parameters:
    -----------
    D_normalized : float
        Diffusion coefficient in normalized units (from PINN output)
    metadata : dict
        Metadata from load_csv_diffusion_data containing scale factors
    pixel_size_um : float, optional
        Pixel size in micrometers
    frame_time_s : float, optional
        Time between frames in seconds
        
    Returns:
    --------
    dict : Contains D in various unit systems
    """
    results = {'D_normalized': D_normalized}
    
    # Convert to pixel²/frame units
    if 'x_scale' in metadata and 't_scale' in metadata:
        # D_normalized is in normalized_length² / normalized_time
        # D_pixel_frame = D_normalized * (x_scale² / t_scale)
        # Assuming x and y scales are similar
        D_pixel_frame = D_normalized * (metadata['x_scale']**2 / metadata['t_scale'])
        results['D_pixel_frame'] = D_pixel_frame
        results['units_pixel_frame'] = 'pixel²/frame'
    
    # Convert to physical units if calibration provided
    if pixel_size_um is not None and frame_time_s is not None:
        # D_physical = D_pixel_frame * (pixel_size_um)² / frame_time_s
        # Result in µm²/s
        D_um2_s = D_pixel_frame * (pixel_size_um**2) / frame_time_s
        results['D_um2_s'] = D_um2_s
        results['units_physical'] = 'µm²/s'
        
        # Also in cm²/s (common in literature)
        D_cm2_s = D_um2_s * 1e-8
        results['D_cm2_s'] = D_cm2_s
    
    return results
