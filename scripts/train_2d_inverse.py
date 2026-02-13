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

import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot

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
    plot_gradient_histograms
)

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
    
    # Handle case where diff_coeff_true might be None (experimental data)
    if data.get('diff_coeff_true') is not None:
        print(f"True D = {data['diff_coeff_true']}")
    else:
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
    diff_coeff_true = data.get('diff_coeff_true')
    if diff_coeff_true is not None:
        print(f"  DIFF_COEFF_TRUE     = {diff_coeff_true}")
    else:
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
    print(f"  Input file          = {config.data.input_file}")
    print(f"  nx={len(x)}, ny={len(y)}, nt={len(t)}")
    print(f"  Domain: x in [{x.min():.4f},{x.max():.4f}], "
          f"y in [{y.min():.4f},{y.max():.4f}], "
          f"t in [{t.min():.4f},{t.max():.4f}]")
    print(f"  N_u (BC/IC points)  = {config.data.n_u}")
    print(f"  N_f (collocation)   = {config.data.n_f}")
    print(f"  N_obs (interior)    = {config.data.n_obs}")
    
    # Metadata (for CSV data)
    if 'metadata' in data:
        meta = data['metadata']
        if 'csv_mode' in meta:
            print(f"  CSV mode            = {meta['csv_mode']}")
    
    # Results
    print("\n--- Results ---")
    if diff_coeff_true is not None:
        print(f"  D_true              = {diff_coeff_true}")
    print(f"  D_learned           = {training_results['diff_coeff_learned']:.6f}")
    
    if diff_coeff_true is not None:
        diff_err = training_results.get('diff_coeff_error', None)
        if diff_err is not None:
            print(f"  D_error             = {diff_err:.6f}")
    
    # Handle optional keys with .get()
    if 'final_loss' in training_results:
        print(f"  Final loss          = {training_results['final_loss']:.6e}")
    if 'training_time' in training_results:
        print(f"  Training time       = {training_results['training_time']:.2f} s")
    
    # Output files
    print(f"\n--- Output Files ---")
    print(f"  Directory: {output_dir}/")
    print(f"    - training_diagnostics_*.pdf")
    print(f"    - gradient_histogram_*.pdf")
    print(f"    - subfigures/ (individual panels)")
    print(f"    - latex_snippets.txt")
    
    print("\n" + "="*70)


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

    # Generate final gradient histogram (after training)
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

    # Build a grid for visualization if not provided by the data loader
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
    diff_coeff_true = None  # Default for experimental data
    
    if is_pointcloud or data.get('usol', None) is None:
        # Skip viz_slices generation - don't create hundreds of time slice images
        print("[viz] Skipping time-slice visualization for experimental data")
    else:
        # Synthetic mode: compare to ground truth on grid
        diff_coeff_true = data.get('diff_coeff_true', None)
        
        x, y, t = data['grid']
        XX, YY = np.meshgrid(x, y, indexing='xy')
        XY = np.stack([XX.ravel(), YY.ravel()], axis=1)
        # assume single time t if scalar, else use first for comparison plotting
        t0 = float(t[0]) if hasattr(t, '__len__') else float(t)
        tt = np.full((XY.shape[0], 1), t0)
        Xq = np.hstack([XY, tt])

        eval_fn = getattr(model, "evaluate_intensity", None) or getattr(model, "predict_observation", None) or model.evaluate
        u_pred = np.asarray(eval_fn(Xq)).reshape(-1,)

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

    # Training diagnostics (diff_coeff_true now guaranteed to exist)
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