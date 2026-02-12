"""
Main training script for 2D diffusion inverse problem with IDW-PINN.

Supports both MAT and CSV data formats via unified loader.

Orchestrates:
- Configuration loading
- Data preparation (auto-selects loader based on file format)
- Model initialization
- Two-phase training (Adam + L-BFGS)
- Visualization and results summary

Usage:
    python scripts/train_2d_inverse.py --config configs/default_2d_inverse.yaml
    python scripts/train_2d_inverse.py --config configs/csv_2d_inverse.yaml
"""
import os
import sys
import argparse
import numpy as np
import tensorflow as tf

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from idw_pinn.config import Config
from idw_pinn.data import load_data
from idw_pinn.models import PINN
from idw_pinn.training import IDWPINNTrainer
from idw_pinn.utils import (
    plot_2d_solution_comparison, 
    plot_training_diagnostics,
    plot_gradient_histograms  # ADD THIS
)

def _as_float(x):
    # Handles tf.Variable, tf.Tensor, numpy scalar, python float, and callables
    if x is None:
        return None
    if callable(x):
        x = x()
    try:
        return float(x.numpy())
    except Exception:
        try:
            return float(x)
        except Exception:
            return None
        
def setup_environment(seed=123):
    """
    Configure reproducibility and TensorFlow environment.
    """
    # Disable oneDNN custom operations
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # Set random seeds for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    print(f"TensorFlow version: {tf.__version__}")


def print_header(config, data):
    """Print formatted header with problem description."""
    print("\n" + "="*70)
    print("2D Diffusion Inverse Problem with IDW Weighting")
    print("="*70)
    print(f"\nConfiguration: {config.data.input_file}")
    
    print("True D = Unknown (experimental data)")
    
    print(f"Initial D guess = {config.physics.diff_coeff_init}")


def print_final_summary(config, data, model, training_results, output_dir='outputs'):
    """
    Print comprehensive final summary.
    """
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    # Configuration
    print("\n--- Configuration ---")
    print(f"  DIFF_COEFF_TRUE     = Unknown")
    print(f"  DIFF_COEFF_INIT     = {config.physics.diff_coeff_init}")
    print(f"  IDW_EMA_BETA        = {config.idw.ema_beta}")
    print(f"  IDW_EPS             = {config.idw.eps}")
    print(f"  IDW_CLAMP           = ({config.idw.clamp_min}, {config.idw.clamp_max})")
    print(f"  WEIGHT_SUM_TARGET   = {config.idw.weight_sum_target}")
    print(f"  FREEZE_BEFORE_LBFGS = {config.idw.freeze_before_lbfgs}")
    print(f"  ADAM_LR             = {config.training.adam_lr}")
    print(f"  ADAM_EPOCHS         = {config.training.adam_epochs}")
    
    # Data
    print("\n--- Data ---")
    x, y, t = data['grid']
    input_file = getattr(getattr(config, 'data', object()), 'input_file', None)
    if input_file is not None:
        print(f"  Input file          = {input_file}")
    print(f"  nx={len(x)}, ny={len(y)}, nt={len(t)}")
    print(f"  Domain: x in [{x.min():.4f},{x.max():.4f}], "
          f"y in [{y.min():.4f},{y.max():.4f}], "
          f"t in [{t.min():.4f},{t.max():.4f}]")

    # Prefer config values if present; otherwise derive from loaded arrays
    n_u_cfg = getattr(getattr(config, 'data', object()), 'n_u', None)
    n_u = int(n_u_cfg) if n_u_cfg is not None else (int(data['X_u_train'].shape[0]) if ('X_u_train' in data and data['X_u_train'] is not None) else 0)

    n_f_cfg = getattr(getattr(config, 'data', object()), 'n_f', None)
    n_f = int(n_f_cfg) if n_f_cfg is not None else (int(data['X_f_train'].shape[0]) if ('X_f_train' in data and data['X_f_train'] is not None) else 0)

    n_obs_cfg = getattr(getattr(config, 'data', object()), 'n_obs', None)
    n_obs = int(n_obs_cfg) if n_obs_cfg is not None else (int(data['X_obs'].shape[0]) if ('X_obs' in data and data['X_obs'] is not None) else 0)

    n_obs_val = int(data['X_obs_val'].shape[0]) if ('X_obs_val' in data and data['X_obs_val'] is not None) else 0

    print(f"  N_u (BC/IC points)  = {n_u}")
    print(f"  N_f (collocation)   = {n_f}")
    print(f"  N_obs (train)       = {n_obs}")
    if n_obs_val > 0:
        print(f"  N_obs (val)         = {n_obs_val}")

    # Metadata (for CSV data)
    if 'metadata' in data:
        meta = data['metadata']
        print("\n--- Data Metadata ---")
        if 'x_range' in meta:
            print(f"  Original X range    = {meta['x_range']}")
            print(f"  Original Y range    = {meta['y_range']}")
            print(f"  Original T range    = {meta['t_range']}")
        if 'intensity_range' in meta:
            print(f"  Intensity range     = {meta['intensity_range']}")
    
    # Network
    print("\n--- Network ---")
    print(f"  Layers              = {config.network.layers}")
    print(f"  Total parameters    = {model.parameters + 1}")  # +1 for diff_coeff
    
    # Training Time
    print("\n--- Training Time ---")
    print(f"  Adam time           = {training_results['adam_time']:.2f}s")
    print(f"  L-BFGS time         = {training_results['lbfgs_time']:.2f}s")
    print(f"  Total time          = {training_results['adam_time'] + training_results['lbfgs_time']:.2f}s")
    print(f"  L-BFGS iterations   = {training_results['lbfgs_results'].nit}")
    print(f"  L-BFGS func evals   = {training_results['lbfgs_results'].nfev}")
    
    lbfgs_msg = training_results['lbfgs_results'].message
    if isinstance(lbfgs_msg, bytes):
        lbfgs_msg = lbfgs_msg.decode()
    print(f"  L-BFGS termination  = {lbfgs_msg}")
    
    # Results
    print("\n--- Results ---")
    print(f"  D_learned           = {training_results['diff_coeff_learned']:.6f}")
    #print(f"  D_error             = {training_results['diff_coeff_error']:.6f} ")
    diff_err = training_results.get('diff_coeff_error', None)

    if diff_err is not None:
        print(f"  D_error             = {diff_err:.6f}")
    else:
        print("  D_error             = NA (experimental)")
    
    print(f"  D_learned           = {training_results['diff_coeff_learned']:.6f}")
    print(f"  D_error             = N/A (no ground truth)")
    print(f"  Relative L2 error   = {training_results['final_error']:.5e}")
    
    # Unit conversion hint for CSV data
    if 'metadata' in data:
        print("\n--- Unit Conversion ---")
        print(f"  D_learned is in normalized units.")
        print(f"  To convert to physical units, use:")
        print(f"    from idw_pinn.data import convert_diffusion_coefficient")
        print(f"    result = convert_diffusion_coefficient(D_learned, metadata, pixel_size_um, frame_time_s)")
    
    # Final IDW Weights
    print("\n--- Final IDW Weights ---")
    final_lam_bc = training_results['history_lbfgs']['lam_bc'][-1] if training_results['history_lbfgs']['lam_bc'] else training_results['history']['lam_bc'][-1]
    final_lam_data = training_results['history_lbfgs']['lam_data'][-1] if training_results['history_lbfgs']['lam_data'] else training_results['history']['lam_data'][-1]
    final_lam_f = training_results['history_lbfgs']['lam_f'][-1] if training_results['history_lbfgs']['lam_f'] else training_results['history']['lam_f'][-1]
    print(f"  lambda_bc           = {final_lam_bc:.6f}")
    print(f"  lambda_data         = {final_lam_data:.6f}")
    print(f"  lambda_f            = {final_lam_f:.6f}")
    
    # Ipred​(x,y,t) = a * uθ​(x,y,t) + b
    # --- I_pred(x,y,t) = a*u_theta(x,y,t) + b ---
    a = getattr(model, "intensity_scale", None)
    b = getattr(model, "intensity_bias", None)
    
    a_val = _as_float(a)
    b_val = _as_float(b)

    print("\n--- I_pred(x,y,t) = a * u_theta(x,y,t) + b ---")
    print(f"a (intensity scale) = {a_val:.6g}" if a_val is not None else "a (intensity scale) = NA")
    print(f"b (intensity bias)  = {b_val:.6g}" if b_val is not None else "b (intensity bias)  = NA")
    
    
    # Output Files
    print("\n--- Output Files ---")
    print(f"  {output_dir}/diff2D_IDW_inverse.png")
    print(f"  {output_dir}/inverse_diagnostics_2D.png")
    
    print("="*70)


def plot_time_slices_with_val(model, data, output_dir, nx=150, ny=150, prefix="pred"):
    """Prediction-only plots for experimental pointcloud data.
    For each time in data['grid'], evaluate model on an (x,y) grid, overlay held-out validation points (if any),
    and save figures as {prefix}_t{time:.1f}.png in output_dir.
    Uses intensity-space prediction if available (evaluate_intensity or predict_observation).
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Build grid if missing
    if data.get("grid", None) is None:
        # derive bounds from available points
        X_list = []
        for k in ["X_obs", "X_obs_val", "X_u_test", "X_f_train"]:
            if k in data and data[k] is not None and len(data[k]) > 0:
                X_list.append(data[k])
        if not X_list:
            raise ValueError("Cannot build grid: no point data found.")
        X_all = np.vstack(X_list)
        x_min, x_max = float(np.min(X_all[:,0])), float(np.max(X_all[:,0]))
        y_min, y_max = float(np.min(X_all[:,1])), float(np.max(X_all[:,1]))
        t_vals = np.unique(X_all[:,2])
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        data["grid"] = (x, y, t_vals)

    x, y, t_vals = data["grid"]
    x = np.asarray(x); y = np.asarray(y); t_vals = np.asarray(t_vals)

    # evaluator
    eval_fn = getattr(model, "evaluate_intensity", None) or getattr(model, "predict_observation", None) or model.evaluate

    # validation points
    Xv = data.get("X_obs_val", None)
    have_val = Xv is not None and len(Xv) > 0

    XX, YY = np.meshgrid(x, y, indexing="xy")
    XY = np.stack([XX.ravel(), YY.ravel()], axis=1)

    for ti in t_vals:
        tt = np.full((XY.shape[0], 1), float(ti))
        Xq = np.hstack([XY, tt])
        pred = eval_fn(Xq)
        pred = np.asarray(pred).reshape(-1,)
        ZZ = pred.reshape(len(y), len(x))

        fig = plt.figure()
        plt.imshow(
            ZZ,
            origin="lower",
            extent=[x.min(), x.max(), y.min(), y.max()],
            aspect="auto",
        )
        plt.colorbar()
        plt.title(f"Prediction (intensity) at t={float(ti):g}")
        plt.xlabel("x"); plt.ylabel("y")

        if have_val:
            Xv_arr = np.asarray(Xv)
            mask = np.isclose(Xv_arr[:,2], float(ti))
            if np.any(mask):
                plt.scatter(Xv_arr[mask,0], Xv_arr[mask,1], s=8)

        fname = os.path.join(output_dir, f"{prefix}_t{float(ti):.1f}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close(fig)
        
def main():
    """Main execution flow."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train 2D diffusion inverse PINN')
    parser.add_argument('--config', type=str, 
                        default='configs/default_2d_inverse.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Directory for output files')
    args = parser.parse_args()
    
    # Setup
    setup_environment(seed=123)
    
    # Load configuration
    config = Config(args.config)
    
    # Load data using unified loader (auto-selects CSV or MAT based on file extension)
    print("\n--- Loading Data ---")
    data = load_data(config)
    
    # Print header after loading data (so we know if diff_coeff_true is available)
    print_header(config, data)
    
    print(f"Loaded {config.data.input_file}")
    if data.get('diff_coeff_true') is not None:
        print(f"  Ground truth D = {data['diff_coeff_true']}")
    print(f"  Training points: BC/IC={data['X_u_train'].shape[0]}, "
          f"Observations={data['X_obs'].shape[0]}, "
          f"Collocation={data['X_f_train'].shape[0]}")
    
    # Create model
    print("\n--- Initializing Model ---")
    lb, ub = data['bounds']
    # Enable a learnable affine mapping from NN output (physical field) to
    # observed intensity when using masked experimental pointcloud CSVs.
    model_cfg = getattr(config, 'model', None)
    intensity_scaling_enabled = bool(getattr(model_cfg, 'intensity_scaling_enabled', False))
    if data.get('metadata', {}).get('csv_mode') == 'pointcloud':
        intensity_scaling_enabled = True
    model = PINN(
        layers=config.network.layers,
        lb=lb,
        ub=ub,
        diff_coeff_init=config.physics.diff_coeff_init,
        idw_config=config,
        intensity_scaling_enabled=intensity_scaling_enabled,
        intensity_scale_init=float(getattr(model_cfg, 'intensity_scale_init', 1.0)),
        intensity_bias_init=float(getattr(model_cfg, 'intensity_bias_init', 0.0)),
    )
    print(f"Network: {config.network.layers}")
    extra = 1  # diff_coeff
    if intensity_scaling_enabled:
        extra += 2  # a_raw, b
    print(f"Total parameters: {model.parameters + extra}")
    if intensity_scaling_enabled:
        print("Observation mapping: I = a*u + b (a>0 via softplus)")
    
    config.output_dir = args.output_dir

    # Create trainer
    print("\n--- Initializing Trainer ---")
    trainer = IDWPINNTrainer(model, config, data)
    
    print("\n--- Initial Gradient Distribution ---")
    plot_gradient_histograms(
        model=model,
        data=data,
        epoch=0,
        weighting_method='Before Training',
        output_dir=args.output_dir
    )

    # Train
    print("\n--- Starting Training ---")
    training_results = trainer.train()

    # ADD THIS BLOCK: Generate final gradient histogram (after training)
    print("\n--- Final Gradient Distribution ---")
    plot_gradient_histograms(
        model=model,
        data=data,
        epoch='final',
        weighting_method='IDW' if config.idw.enabled else 'Uniform',
        output_dir=args.output_dir
    )
    
    # Generate visualizations
    print("\n--- Generating Visualizations ---")
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Build a grid for visualization if not provided by the data loader
    # ------------------------------------------------------------------
    if data.get('grid', None) is None:
        X_list = []
        for k in ['X_obs', 'X_obs_val', 'X_u_test', 'X_f_train']:
            if k in data and data[k] is not None and len(data[k]) > 0:
                X_list.append(data[k])
        if len(X_list) == 0:
            raise ValueError("Cannot build visualization grid: no point arrays found.")
        X_all = np.vstack(X_list)

        x_min, x_max = float(np.min(X_all[:, 0])), float(np.max(X_all[:, 0]))
        y_min, y_max = float(np.min(X_all[:, 1])), float(np.max(X_all[:, 1]))
        t_vals = np.unique(X_all[:, 2])

        nx = 150
        ny = 150
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        t = t_vals
        data['grid'] = (x, y, t)

    # Experimental/pointcloud mode: no ground-truth solution grid
    is_pointcloud = (data.get('metadata', {}).get('csv_mode') == 'pointcloud')
    if is_pointcloud or data.get('usol', None) is None:
        viz_dir = os.path.join(args.output_dir, "viz_slices")
        plot_time_slices_with_val(model, data, viz_dir, nx=150, ny=150, prefix="pred")
        print(f"[viz] Saved prediction-only slices to: {viz_dir}")
        diff_coeff_true = None
    else:
        # Synthetic mode: compare to ground truth on grid
        x, y, t = data['grid']
        XX, YY = np.meshgrid(x, y, indexing='xy')
        XY = np.stack([XX.ravel(), YY.ravel()], axis=1)
        # assume single time t if scalar, else use first for comparison plotting
        t0 = float(t[0]) if hasattr(t, '__len__') else float(t)
        tt = np.full((XY.shape[0], 1), t0)
        Xq = np.hstack([XY, tt])

        eval_fn = getattr(model, "evaluate_intensity", None) or getattr(model, "predict_observation", None) or model.evaluate
        u_pred = np.asarray(eval_fn(Xq)).reshape(-1,)

        diff_coeff_true = data.get('diff_coeff_true', None)

        plot_2d_solution_comparison(
            u_pred=u_pred,
            usol=data['usol'],
            x=x,
            y=y,
            t=t0,
            diff_coeff_learned=training_results['diff_coeff_learned'],
            diff_coeff_true=diff_coeff_true,
            output_dir=args.output_dir
        )

    # Training diagnostics
    plot_training_diagnostics(
        history_adam=training_results['history'],
        history_lbfgs=training_results['history_lbfgs'],
        diff_coeff_true=diff_coeff_true,
        output_dir=args.output_dir
    )

    # Print comprehensive summary
    print_final_summary(config, data, model, training_results, args.output_dir)


if __name__ == "__main__":
    main()