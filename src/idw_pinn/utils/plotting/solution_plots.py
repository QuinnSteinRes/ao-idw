"""
Solution comparison plots for 2D diffusion inverse problems.

Visualizes predicted vs true/measured solutions at multiple time points
with error analysis. Supports both numerical (with ground truth) and 
experimental (with target range) data types.

Updated: January 2026
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from .styles import (set_publication_style, FONT_CONFIG, DPI_SAVE, 
                     DATA_CONFIG, DATA_TYPE)
from .utils import generate_unique_filename, ensure_output_dirs, append_latex_snippet


def _save_single_panel(X, Y, data, cmap, vmin, vmax, title, cbar_label, filepath):
    """Save a single panel as a standalone figure for LaTeX."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.pcolormesh(X, Y, data, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=FONT_CONFIG['title'])
    ax.set_xlabel('x', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel('y', fontsize=FONT_CONFIG['axis_label'])
    ax.set_aspect('equal')
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=FONT_CONFIG['colorbar_label'])
    cbar.ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=DPI_SAVE, bbox_inches='tight')
    plt.close()


def plot_2d_solution_comparison(u_pred, usol, x, y, t, diff_coeff_learned,
                                 diff_coeff_true=None, data_type=None,
                                 output_dir='outputs', filename=None,
                                 save_subfigures=True):
    """
    Plot 2D solution comparison at 4 time slices with PERCENT ERROR.
    
    Creates a 3-row figure:
    - Row 0: True/measured solution at 4 time points
    - Row 1: Predicted solution at 4 time points
    - Row 2: Percent error at 4 time points
    
    Args:
        u_pred: Predicted solution, shape (Nx+1, Ny+1, Nt) or flattened
        usol: True/measured solution, shape (Nx+1, Ny+1, Nt)
        x: x coordinates (Nx+1,)
        y: y coordinates (Ny+1,)
        t: time coordinates (Nt,)
        diff_coeff_learned: Learned diffusion coefficient
        diff_coeff_true: True D (for numerical) or None (for experimental)
        data_type: 'numerical' or 'experimental'
        output_dir: Directory to save figures
        filename: Output filename (if None, generates unique name)
        save_subfigures: Whether to save individual subfigures for LaTeX
    
    Returns:
        dict: Paths to saved files {'whole': path, 'subfigures': [paths]}
    """
    # Use global DATA_TYPE if not specified
    if data_type is None:
        data_type = DATA_TYPE
        
    set_publication_style()
    subfig_dir, run_id = ensure_output_dirs(output_dir)
    
    # Generate unique filename if not provided
    if filename is None:
        filename = generate_unique_filename('solution_comparison', 'pdf', diff_coeff_learned)
    
    # Reshape prediction to match solution shape if needed
    if u_pred.ndim == 2 and u_pred.shape[1] == 1:
        u_pred = u_pred.flatten()
    
    if u_pred.ndim == 1:
        u_pred_reshaped = np.reshape(u_pred, usol.shape, order='C')
    else:
        u_pred_reshaped = u_pred
    
    # Select 4 time indices for plotting
    nt = len(t)
    t_indices = [0, nt//3, 2*nt//3, nt-1]
    n_cols = len(t_indices)
    
    # Create meshgrid
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Compute global colorbar limits for solution
    u_vmin = min(usol.min(), u_pred_reshaped.min())
    u_vmax = max(usol.max(), u_pred_reshaped.max())
    
    # Compute PERCENT ERROR
    # Avoid division by zero: use max(|true|, small_value) as denominator
    usol_abs_max = np.maximum(np.abs(usol), 1e-10)
    percent_error_all = 100.0 * np.abs(usol - u_pred_reshaped) / usol_abs_max
    
    # Cap percent error for visualization (very small true values cause huge %)
    percent_error_all = np.clip(percent_error_all, 0, 100)
    err_vmax = min(percent_error_all.max(), 50)  # Cap at 50% for colorbar
    
    # Determine labels based on data type
    if data_type == 'numerical':
        true_label = 'True'
        d_display = f'True D={DATA_CONFIG["numerical"]["diff_coeff_true"]}'
    else:
        true_label = 'Measured'
        d_range = DATA_CONFIG['experimental']['diff_coeff_range']
        d_display = f'Target D: {d_range[0]:.4f}-{d_range[1]:.4f}'
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(18, 13))
    gs = gridspec.GridSpec(3, n_cols + 1, width_ratios=[1, 1, 1, 1, 0.05],
                           wspace=0.25, hspace=0.3)
    
    saved_subfigures = []
    
    # Row 0: True/measured solution
    axes_true = [fig.add_subplot(gs[0, i]) for i in range(n_cols)]
    for i, ti in enumerate(t_indices):
        im0 = axes_true[i].pcolormesh(X, Y, usol[:, :, ti], cmap='jet',
                                       shading='auto', vmin=u_vmin, vmax=u_vmax)
        axes_true[i].set_title(f'{true_label}, t={t[ti]:.3f}s', fontsize=FONT_CONFIG['title'])
        axes_true[i].set_xlabel('x', fontsize=FONT_CONFIG['axis_label'])
        if i == 0:
            axes_true[i].set_ylabel('y', fontsize=FONT_CONFIG['axis_label'])
        axes_true[i].set_aspect('equal')
        axes_true[i].tick_params(labelsize=FONT_CONFIG['tick_label'])
        
        # Save subfigure (clean filename without D value)
        if save_subfigures:
            subfig_path = os.path.join(subfig_dir, f'{true_label.lower()}_t{i}.pdf')
            _save_single_panel(X, Y, usol[:, :, ti], 'jet', u_vmin, u_vmax,
                              f'{true_label}, t={t[ti]:.3f}s', 'u', subfig_path)
            saved_subfigures.append(subfig_path)
    
    cax0 = fig.add_subplot(gs[0, n_cols])
    cbar0 = fig.colorbar(im0, cax=cax0)
    cbar0.set_label(f'u ({true_label})', fontsize=FONT_CONFIG['colorbar_label'])
    cbar0.ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    # Row 1: Predicted solution
    axes_pred = [fig.add_subplot(gs[1, i]) for i in range(n_cols)]
    for i, ti in enumerate(t_indices):
        im1 = axes_pred[i].pcolormesh(X, Y, u_pred_reshaped[:, :, ti], cmap='jet',
                                       shading='auto', vmin=u_vmin, vmax=u_vmax)
        axes_pred[i].set_title(f'Predicted, t={t[ti]:.3f}s', fontsize=FONT_CONFIG['title'])
        axes_pred[i].set_xlabel('x', fontsize=FONT_CONFIG['axis_label'])
        if i == 0:
            axes_pred[i].set_ylabel('y', fontsize=FONT_CONFIG['axis_label'])
        axes_pred[i].set_aspect('equal')
        axes_pred[i].tick_params(labelsize=FONT_CONFIG['tick_label'])
        
        # Save subfigure (clean filename)
        if save_subfigures:
            subfig_path = os.path.join(subfig_dir, f'pred_t{i}.pdf')
            _save_single_panel(X, Y, u_pred_reshaped[:, :, ti], 'jet', u_vmin, u_vmax,
                              f'Predicted, t={t[ti]:.3f}s', 'u', subfig_path)
            saved_subfigures.append(subfig_path)
    
    cax1 = fig.add_subplot(gs[1, n_cols])
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.set_label('u (Pred)', fontsize=FONT_CONFIG['colorbar_label'])
    cbar1.ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    # Row 2: Percent Error
    axes_err = [fig.add_subplot(gs[2, i]) for i in range(n_cols)]
    for i, ti in enumerate(t_indices):
        im2 = axes_err[i].pcolormesh(X, Y, percent_error_all[:, :, ti], cmap='hot',
                                      shading='auto', vmin=0, vmax=err_vmax)
        axes_err[i].set_title(f'% Error, t={t[ti]:.3f}s', fontsize=FONT_CONFIG['title'])
        axes_err[i].set_xlabel('x', fontsize=FONT_CONFIG['axis_label'])
        if i == 0:
            axes_err[i].set_ylabel('y', fontsize=FONT_CONFIG['axis_label'])
        axes_err[i].set_aspect('equal')
        axes_err[i].tick_params(labelsize=FONT_CONFIG['tick_label'])
        
        # Save subfigure (clean filename)
        if save_subfigures:
            subfig_path = os.path.join(subfig_dir, f'error_t{i}.pdf')
            _save_single_panel(X, Y, percent_error_all[:, :, ti], 'hot', 0, err_vmax,
                              f'% Error, t={t[ti]:.3f}s', '% Error', subfig_path)
            saved_subfigures.append(subfig_path)
    
    cax2 = fig.add_subplot(gs[2, n_cols])
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.set_label('% Error', fontsize=FONT_CONFIG['colorbar_label'])
    cbar2.ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    # Title with appropriate D display
    plt.suptitle(f'2D Diffusion: Learned D = {diff_coeff_learned:.6f} ({d_display})',
                 fontsize=FONT_CONFIG['suptitle'])
    
    # Save as PDF only
    filepath = os.path.join(output_dir, filename.replace('.png', '.pdf'))
    plt.savefig(filepath, dpi=DPI_SAVE, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()
    
    # Generate LaTeX snippet
    if save_subfigures and saved_subfigures:
        append_latex_snippet(
            output_dir=output_dir,
            subfigure_paths=saved_subfigures,
            figure_type='solution_comparison',
            caption=f'2D diffusion solution comparison. Learned D = {diff_coeff_learned:.6f}.',
            label='fig:solution_comparison',
            run_id=run_id
        )
    
    return {'whole': filepath, 'subfigures': saved_subfigures}


def plot_experimental_comparison(u_pred, intensity_measured, x, y, t,
                                  diff_coeff_learned, output_dir='outputs',
                                  filename=None, roi_bounds=None,
                                  X_obs=None, save_subfigures=True):
    """
    Plot comparison for experimental data (intensity-based, no ground truth D).
    
    Similar to plot_2d_solution_comparison but tailored for experimental data:
    - Uses "Measured" instead of "True"
    - Shows target D range instead of single true value
    - Optionally overlays ROI boundary and observation points
    
    Args:
        u_pred: Predicted intensity
        intensity_measured: Measured intensity from experiment
        x, y, t: Coordinate arrays
        diff_coeff_learned: Learned D (normalized units)
        output_dir: Directory to save figures
        filename: Output filename (if None, generates unique name)
        roi_bounds: Optional dict with 'x_min', 'x_max', 'y_min', 'y_max' for ROI overlay
        X_obs: Observation points (N, 3) for overlay
        save_subfigures: Whether to save individual subfigures
    
    Returns:
        dict: Paths to saved files
    """
    return plot_2d_solution_comparison(
        u_pred=u_pred,
        usol=intensity_measured,
        x=x, y=y, t=t,
        diff_coeff_learned=diff_coeff_learned,
        diff_coeff_true=None,
        data_type='experimental',
        output_dir=output_dir,
        filename=filename,
        save_subfigures=save_subfigures
    )