"""
Visualization utilities for 2D diffusion inverse problems.

Provides publication-ready plots with:
- Large fonts/axes for Overleaf/presentations
- Unique identifiers (config params + datetime) in filenames
- Whole figures + separate subfigures for LaTeX integration
- Support for numerical (with true D=0.2) and experimental data (target D range)
- Percent error display instead of absolute error
- Consistent axis limits across runs for comparison
- Both LOG and LINEAR scale versions of D evolution and lambda plots

Updated: January 2026
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import os
from datetime import datetime
import uuid


# =============================================================================
# DATA TYPE SELECTION - CHANGE THIS FLAG
# =============================================================================

DATA_TYPE = 'numerical'  # OPTIONS: 'numerical' or 'experimental'


# =============================================================================
# PUBLICATION-READY DEFAULTS
# =============================================================================

FONT_CONFIG = {
    'title': 16,
    'suptitle': 18,
    'axis_label': 14,
    'tick_label': 12,
    'colorbar_label': 13,
    'legend': 12,
}

# Hardcoded axis limits for cross-run consistency (all log scale)
AXIS_LIMITS = {
    #'D_evolution': (1e-5, 1),
    'D_evolution': (1e-8, 1),
    'D_error': (1e-5, 1e5),
    'losses': (1e-12, 1e2),
    'lambdas': (1e-3, 1e5),
}

DPI_SAVE = 300

# Data type configurations
DATA_CONFIG = {
    'numerical': {
        'diff_coeff_true': 0.2,
        'diff_coeff_display': '0.2',
        'label': 'True D',
    },
    'experimental': {
        # Target D range in m²/s: 3.15e-10 to 4.05e-10
        # Corresponding D_norm range: 0.000507 to 0.000652
        'diff_coeff_range': (0.000507, 0.000652),
        'diff_coeff_physical_range': (3.15e-10, 4.05e-10),  # m²/s
        'label': 'Target D range',
    }
}


def set_publication_style():
    """Set matplotlib defaults for publication-quality figures."""
    plt.rcParams.update({
        'font.size': FONT_CONFIG['axis_label'],
        'axes.titlesize': FONT_CONFIG['title'],
        'axes.labelsize': FONT_CONFIG['axis_label'],
        'xtick.labelsize': FONT_CONFIG['tick_label'],
        'ytick.labelsize': FONT_CONFIG['tick_label'],
        'legend.fontsize': FONT_CONFIG['legend'],
        'figure.titlesize': FONT_CONFIG['suptitle'],
        'axes.linewidth': 1.2,
        'lines.linewidth': 1.5,
        'savefig.dpi': DPI_SAVE,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


# =============================================================================
# FILENAME GENERATION
# =============================================================================

# Module-level run ID - generated once per import/session
_SESSION_RUN_ID = None


def _get_session_run_id():
    """Get or create a session-wide unique run ID."""
    global _SESSION_RUN_ID
    if _SESSION_RUN_ID is None:
        timestamp = datetime.now().strftime('%d%m%y_%H%M%S')
        short_uuid = uuid.uuid4().hex[:4]
        _SESSION_RUN_ID = f"run_{timestamp}_{short_uuid}"
    return _SESSION_RUN_ID


def reset_session_run_id():
    """Reset the session run ID (call at start of new training run)."""
    global _SESSION_RUN_ID
    _SESSION_RUN_ID = None


def generate_unique_filename(base_name: str, extension: str = 'png', 
                             diff_coeff: float = None) -> str:
    """
    Generate unique filename with timestamp and UUID.
    
    Format: {base_name}_{timestamp}_{uuid}.{extension}
    Note: D value no longer included in filename.
    """
    timestamp = datetime.now().strftime('%d%m%y_%H%M%S')
    short_uuid = uuid.uuid4().hex[:4]
    return f"{base_name}_{timestamp}_{short_uuid}.{extension}"


def _ensure_dirs(output_dir: str):
    """Ensure output and subfigures directories exist with session run ID."""
    os.makedirs(output_dir, exist_ok=True)
    run_id = _get_session_run_id()
    subfig_dir = os.path.join(output_dir, 'subfigures', run_id)
    os.makedirs(subfig_dir, exist_ok=True)
    return subfig_dir, run_id


# =============================================================================
# LATEX SNIPPET GENERATION
# =============================================================================

def _append_latex_snippet(output_dir: str, subfigure_paths: list, 
                          figure_type: str, caption: str = '', label: str = '',
                          run_id: str = None):
    """
    Append LaTeX snippet for subfigures to latex_snippets.txt.
    
    Args:
        output_dir: Directory containing the output
        subfigure_paths: List of paths to subfigure PDFs
        figure_type: Type of figure (e.g., 'solution_comparison', 'training_diagnostics')
        caption: Figure caption text
        label: LaTeX label for the figure
        run_id: Shared run ID for this session
    """
    latex_file = os.path.join(output_dir, 'latex_snippets.txt')
    
    if run_id is None:
        run_id = _get_session_run_id()
    
    # Extract just filenames (relative to IDW/Figs/<run_id>/ folder in Overleaf)
    filenames = [os.path.basename(p) for p in subfigure_paths]
    
    # Determine number of columns based on figure type
    # - solution_comparison: 4 columns (12 subfigures in 3 rows of 4)
    # - training_diagnostics: 3 columns (6 subfigures in 2 rows of 3)
    if figure_type == 'training_diagnostics':
        n_cols = min(3, len(filenames))
    else:
        n_cols = min(4, len(filenames))
    
    # Generate timestamp for this entry
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Build LaTeX snippet
    tabular_cols = 'c' * n_cols
    snippet = f"""
% =============================================================================
% {figure_type.upper()} - Generated: {timestamp}
% OVERLEAF DIRECTORY: IDW/Figs/{run_id}
% Upload all subfigures from outputs/subfigures/{run_id}/ to this directory.
% =============================================================================
\\begin{{figure}}[htbp]
\\centering
\\setlength{{\\tabcolsep}}{{2pt}}
\\begin{{tabular}}{{{tabular_cols}}}
"""
    
    # Labels based on number of subfigures
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', 
              '(i)', '(j)', '(k)', '(l)']
    
    for row_start in range(0, len(filenames), n_cols):
        row_files = filenames[row_start:row_start + n_cols]
        row_labels = labels[row_start:row_start + len(row_files)]
        
        # Image row
        img_commands = [f'\\includegraphics[width={0.24}\\textwidth]{{IDW/Figs/{run_id}/{f}}}' 
                       for f in row_files]
        snippet += '    ' + ' &\n    '.join(img_commands) + r' \\' + '\n'
        
        # Label row
        label_commands = [f'\\small {lbl}' for lbl in row_labels]
        snippet += '    ' + ' & '.join(label_commands) + r' \\[0.5em]' + '\n'
    
    snippet += f"""\\end{{tabular}}
\\caption{{{caption if caption else f'{figure_type} results'}}}
\\label{{{label if label else f'fig:{figure_type}'}}}
\\end{{figure}}

% Upload these files to IDW/Figs/{run_id}/:
"""
    for p in subfigure_paths:
        snippet += f"%   {os.path.basename(p)}\n"
    
    snippet += "\n"
    
    # Append to file
    with open(latex_file, 'a') as f:
        f.write(snippet)
    
    print(f"LaTeX snippet appended to: {latex_file}")
    print(f">>> Upload subfigures to Overleaf: IDW/Figs/{run_id}/")


# =============================================================================
# SOLUTION COMPARISON PLOTS
# =============================================================================

def plot_2d_solution_comparison(u_pred, usol, x, y, t, diff_coeff_learned,
                                 diff_coeff_true=None, data_type=None,
                                 output_dir='outputs', filename=None,
                                 save_subfigures=True):
    # Use global DATA_TYPE if not specified
    if data_type is None:
        data_type = DATA_TYPE
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
    set_publication_style()
    subfig_dir, run_id = _ensure_dirs(output_dir)
    
    # Generate unique filename if not provided
    if filename is None:
        filename = generate_unique_filename('solution_comparison', 'pdf', diff_coeff_learned)
    
    # Reshape prediction to match solution shape if needed
    if u_pred.ndim == 2 and u_pred.shape[1] == 1:
        u_pred = u_pred.flatten()
    
    
    # -------------------------------------------------------------------------
    # Experimental / pointcloud mode: no ground-truth grid provided (usol is None)
    # Fall back to prediction-only plots on the provided (x, y, t) grid.
    # -------------------------------------------------------------------------
    if usol is None:
        os.makedirs(output_dir, exist_ok=True)

        nx = len(x)
        ny = len(y)
        nt = len(t)

        u_arr = np.asarray(u_pred)

        # Try to reshape u_pred into a 3D grid [ny, nx, nt] (or [nx, ny, nt])
        if u_arr.ndim == 1:
            if u_arr.size == nx * ny * nt:
                # Prefer [ny, nx, nt] for imshow with extent=[x_min,x_max,y_min,y_max]
                try:
                    u_grid = np.reshape(u_arr, (ny, nx, nt), order='C')
                except Exception:
                    u_grid = np.reshape(u_arr, (nx, ny, nt), order='C').transpose(1, 0, 2)
            elif u_arr.size == (nx * ny):
                # Single time slice only
                u_grid = np.reshape(u_arr, (ny, nx, 1), order='C')
                nt = 1
            else:
                raise ValueError(
                    f"u_pred has size {u_arr.size}, but expected nx*ny*nt={nx*ny*nt} (or nx*ny={nx*ny})."
                )
        elif u_arr.ndim == 2:
            # Assume already [ny, nx] single slice
            u_grid = u_arr.reshape(ny, nx, 1)
            nt = 1
        elif u_arr.ndim == 3:
            u_grid = u_arr
        else:
            u_grid = np.reshape(u_arr, (ny, nx, nt), order='C')

        # Decide which time indices to plot: all if <= 6, else 4 evenly spaced.
        if nt <= 6:
            time_ids = list(range(nt))
        else:
            time_ids = [0, nt // 3, (2 * nt) // 3, nt - 1]

        for k in time_ids:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            im = ax.imshow(
                u_grid[:, :, k],
                origin='lower',
                extent=[float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))],
                aspect='auto'
            )
            fig.colorbar(im, ax=ax)
            ax.set_title(f"Prediction (no ground truth) at t={float(t[k]):g}")
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            fig.tight_layout()

            fname = filename if (filename is not None and nt == 1) else f"pred_only_t{float(t[k]):.1f}.png"
            fig.savefig(os.path.join(output_dir, fname), dpi=200)
            plt.close(fig)

        return
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
        _append_latex_snippet(
            output_dir=output_dir,
            subfigure_paths=saved_subfigures,
            figure_type='solution_comparison',
            caption=f'2D diffusion solution comparison. Learned D = {diff_coeff_learned:.6f}.',
            label='fig:solution_comparison',
            run_id=run_id
        )
    
    return {'whole': filepath, 'subfigures': saved_subfigures}


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


# =============================================================================
# TRAINING DIAGNOSTICS
# =============================================================================

def plot_training_diagnostics(history_adam, history_lbfgs, diff_coeff_true=None,
                               data_type=None, output_dir='outputs',
                               filename=None, save_subfigures=True):
    """
    Plot comprehensive training diagnostics in 2x3 layout.
    
    Layout:
    - Row 0: D evolution (log), D evolution (LINEAR), D error (log)
    - Row 1: Loss components (log), Lambda weights (log), Lambda weights (LINEAR)
    
    Creates diagnostic plots showing:
    - D evolution in both log and linear scale
    - D error evolution (log scale)
    - Loss components (BC/IC, Data, PDE) over epochs (log scale)
    - Lambda weights in both log and linear scale
    
    Args:
        history_adam: Dict with keys 'diff_coeff', 'loss_bc', 'loss_data', 'loss_f',
                      'lam_bc', 'lam_data', 'lam_f' from Adam phase
        history_lbfgs: Dict with same keys from L-BFGS phase
        diff_coeff_true: True D for numerical data, or None for experimental
        data_type: 'numerical' or 'experimental'
        output_dir: Directory to save figure
        filename: Output filename (if None, generates unique name)
        save_subfigures: Whether to save individual subfigures for LaTeX
    
    Returns:
        dict: Paths to saved files
    """
    # Use global DATA_TYPE if not specified
    if data_type is None:
        data_type = DATA_TYPE
        
    set_publication_style()
    subfig_dir, run_id = _ensure_dirs(output_dir)
    
    # Generate unique filename if not provided
    if filename is None:
        final_D = history_lbfgs.get('diff_coeff', history_adam['diff_coeff'])[-1] \
                  if history_lbfgs.get('diff_coeff') else history_adam['diff_coeff'][-1]
        filename = generate_unique_filename('training_diagnostics', 'pdf', final_D)
    
    # Combine histories
    adam_epochs = len(history_adam['diff_coeff'])
    
    diff_all = list(history_adam['diff_coeff']) + list(history_lbfgs.get('diff_coeff', []))
    loss_bc_all = list(history_adam['loss_bc']) + list(history_lbfgs.get('loss_bc', []))
    loss_data_all = list(history_adam['loss_data']) + list(history_lbfgs.get('loss_data', []))
    loss_f_all = list(history_adam['loss_f']) + list(history_lbfgs.get('loss_f', []))
    lam_bc_all = list(history_adam['lam_bc']) + list(history_lbfgs.get('lam_bc', []))
    lam_data_all = list(history_adam['lam_data']) + list(history_lbfgs.get('lam_data', []))
    lam_f_all = list(history_adam['lam_f']) + list(history_lbfgs.get('lam_f', []))
    
    total_iterations = len(diff_all)
    
    # Determine D reference display
    if data_type == 'numerical':
        d_true = diff_coeff_true if diff_coeff_true is not None else DATA_CONFIG['numerical']['diff_coeff_true']
        d_label = f'True D={d_true}'
        d_range = None
    else:
        d_range = DATA_CONFIG['experimental']['diff_coeff_range']
        d_label = f'Target: {d_range[0]:.4f}-{d_range[1]:.4f}'
        d_true = None
    
    # Create 2x3 figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    saved_subfigures = []
    
    # -------------------------------------------------------------------------
    # Plot 1: D evolution (LOG scale)
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    ax.semilogy(range(adam_epochs), history_adam['diff_coeff'], 'b-', alpha=0.8, linewidth=1.5, label='Adam')
    if history_lbfgs.get('diff_coeff'):
        ax.semilogy(range(adam_epochs, total_iterations), history_lbfgs['diff_coeff'],
                'b--', alpha=0.8, linewidth=1.5, label='L-BFGS')
    
    # Reference line(s)
    if d_true is not None:
        ax.axhline(y=d_true, color='r', linestyle='-', linewidth=2, label=d_label)
    elif d_range is not None:
        ax.axhspan(d_range[0], d_range[1], alpha=0.3, color='green', label=d_label)
    
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, linewidth=1.5, label='Adam→L-BFGS')
    ax.set_ylim(AXIS_LIMITS['D_evolution'])
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel('D (log)', fontsize=FONT_CONFIG['axis_label'])
    ax.set_title('D Evolution (Log Scale)', fontsize=FONT_CONFIG['title'])
    ax.legend(fontsize=FONT_CONFIG['legend'])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    if save_subfigures:
        path = os.path.join(subfig_dir, 'D_evolution_log.pdf')
        _save_diagnostic_subplot_direct(
            x_data=[(range(adam_epochs), history_adam['diff_coeff'], 'b-', 'Adam'),
                    (range(adam_epochs, total_iterations), history_lbfgs.get('diff_coeff', []), 'b--', 'L-BFGS')],
            ylabel='D (log)', title='D Evolution (Log Scale)',
            ylim=AXIS_LIMITS['D_evolution'], adam_epochs=adam_epochs,
            d_true=d_true, d_range=d_range, d_label=d_label,
            filepath=path, use_log=True
        )
        saved_subfigures.append(path)
    
    # -------------------------------------------------------------------------
    # Plot 2: D evolution (LINEAR scale) - replaces duplicate log plot
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    ax.plot(range(adam_epochs), history_adam['diff_coeff'], 'b-', alpha=0.8, linewidth=1.5, label='Adam')
    if history_lbfgs.get('diff_coeff'):
        ax.plot(range(adam_epochs, total_iterations), history_lbfgs['diff_coeff'],
                'b--', alpha=0.8, linewidth=1.5, label='L-BFGS')
    
    if d_true is not None:
        ax.axhline(y=d_true, color='r', linestyle='-', linewidth=2, label=d_label)
    elif d_range is not None:
        ax.axhspan(d_range[0], d_range[1], alpha=0.3, color='green', label=d_label)
    
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, linewidth=1.5, label='Adam→L-BFGS')
    # Auto-scale y-axis for linear plot
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel('D', fontsize=FONT_CONFIG['axis_label'])
    ax.set_title('D Evolution (Linear Scale)', fontsize=FONT_CONFIG['title'])
    ax.legend(fontsize=FONT_CONFIG['legend'])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    if save_subfigures:
        path = os.path.join(subfig_dir, 'D_evolution_linear.pdf')
        _save_diagnostic_subplot_direct(
            x_data=[(range(adam_epochs), history_adam['diff_coeff'], 'b-', 'Adam'),
                    (range(adam_epochs, total_iterations), history_lbfgs.get('diff_coeff', []), 'b--', 'L-BFGS')],
            ylabel='D', title='D Evolution (Linear Scale)',
            ylim=None, adam_epochs=adam_epochs,
            d_true=d_true, d_range=d_range, d_label=d_label,
            filepath=path, use_log=False
        )
        saved_subfigures.append(path)
    
    # -------------------------------------------------------------------------
    # Plot 3: D error evolution (numerical) or % error from midpoint (experimental)
    # -------------------------------------------------------------------------
    ax = axes[0, 2]
    
    if data_type == 'numerical' and d_true is not None:
        d_error = np.abs(np.array(diff_all) - d_true)
        ax.semilogy(range(adam_epochs), d_error[:adam_epochs], 'b-', alpha=0.8, linewidth=1.5)
        if len(d_error) > adam_epochs:
            ax.semilogy(range(adam_epochs, total_iterations), d_error[adam_epochs:],
                        'b--', alpha=0.8, linewidth=1.5)
        ylabel = '|D - D_true|'
        title = 'D Error Evolution'
        error_data = d_error
    else:
        # For experimental: percent error from target range midpoint
        d_arr = np.array(diff_all)
        d_low, d_high = d_range
        d_midpoint = (d_low + d_high) / 2.0
        percent_error = np.abs(d_arr - d_midpoint) / d_midpoint * 100.0
        ax.semilogy(range(adam_epochs), percent_error[:adam_epochs] + 1e-12, 'b-', alpha=0.8, linewidth=1.5)
        if len(percent_error) > adam_epochs:
            ax.semilogy(range(adam_epochs, total_iterations), percent_error[adam_epochs:] + 1e-12,
                        'b--', alpha=0.8, linewidth=1.5)
        ylabel = '% Error from midpoint'
        title = 'D % Error from Target Midpoint'
        error_data = percent_error + 1e-12
    
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, linewidth=1.5, label='Adam→L-BFGS')
    ax.set_ylim(AXIS_LIMITS['D_error'])
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel(ylabel, fontsize=FONT_CONFIG['axis_label'])
    ax.set_title(title, fontsize=FONT_CONFIG['title'])
    ax.legend(fontsize=FONT_CONFIG['legend'])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    if save_subfigures:
        path = os.path.join(subfig_dir, 'D_error.pdf')
        _save_diagnostic_subplot_direct(
            x_data=[(range(adam_epochs), error_data[:adam_epochs], 'b-', 'Adam'),
                    (range(adam_epochs, total_iterations), error_data[adam_epochs:] if len(error_data) > adam_epochs else [], 'b--', 'L-BFGS')],
            ylabel=ylabel, title=title,
            ylim=AXIS_LIMITS['D_error'], adam_epochs=adam_epochs,
            d_true=None, d_range=None, d_label=None,
            filepath=path, use_log=True
        )
        saved_subfigures.append(path)
    
    # -------------------------------------------------------------------------
    # Plot 4: Loss components (LOG scale)
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    ax.semilogy(range(adam_epochs), loss_bc_all[:adam_epochs], 'b-', alpha=0.8, linewidth=1.5)
    ax.semilogy(range(adam_epochs), loss_data_all[:adam_epochs], 'orange', alpha=0.8, linewidth=1.5)
    ax.semilogy(range(adam_epochs), loss_f_all[:adam_epochs], 'g-', alpha=0.8, linewidth=1.5)
    if history_lbfgs.get('loss_bc'):
        ax.semilogy(range(adam_epochs, total_iterations), loss_bc_all[adam_epochs:],
                    'b--', alpha=0.8, linewidth=1.5)
        ax.semilogy(range(adam_epochs, total_iterations), loss_data_all[adam_epochs:],
                    'orange', linestyle='--', alpha=0.8, linewidth=1.5)
        ax.semilogy(range(adam_epochs, total_iterations), loss_f_all[adam_epochs:],
                    'g--', alpha=0.8, linewidth=1.5)
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, linewidth=1.5, label='Adam→L-BFGS')
    ax.plot([], [], 'b-', linewidth=2, label=r'$\mathcal{L}_{BC/IC}$')
    ax.plot([], [], 'orange', linewidth=2, label=r'$\mathcal{L}_{data}$')
    ax.plot([], [], 'g-', linewidth=2, label=r'$\mathcal{L}_{PDE}$')
    ax.set_ylim(AXIS_LIMITS['losses'])
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel('Loss', fontsize=FONT_CONFIG['axis_label'])
    ax.set_title('Loss Components (Log Scale)', fontsize=FONT_CONFIG['title'])
    ax.legend(fontsize=FONT_CONFIG['legend'])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    if save_subfigures:
        path = os.path.join(subfig_dir, 'losses.pdf')
        _save_loss_subplot(
            adam_epochs=adam_epochs, total_iterations=total_iterations,
            loss_bc=loss_bc_all, loss_data=loss_data_all, loss_f=loss_f_all,
            ylabel='Loss', title='Loss Components (Log Scale)',
            ylim=AXIS_LIMITS['losses'], filepath=path
        )
        saved_subfigures.append(path)
    
    # -------------------------------------------------------------------------
    # Plot 5: Lambda weights (LOG scale)
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    ax.semilogy(range(adam_epochs), lam_bc_all[:adam_epochs], 'b-', alpha=0.8, linewidth=1.5)
    ax.semilogy(range(adam_epochs), lam_data_all[:adam_epochs], 'orange', alpha=0.8, linewidth=1.5)
    ax.semilogy(range(adam_epochs), lam_f_all[:adam_epochs], 'g-', alpha=0.8, linewidth=1.5)
    if history_lbfgs.get('lam_bc'):
        ax.semilogy(range(adam_epochs, total_iterations), lam_bc_all[adam_epochs:],
                'b--', alpha=0.8, linewidth=1.5)
        ax.semilogy(range(adam_epochs, total_iterations), lam_data_all[adam_epochs:],
                'orange', linestyle='--', alpha=0.8, linewidth=1.5)
        ax.semilogy(range(adam_epochs, total_iterations), lam_f_all[adam_epochs:],
                'g--', alpha=0.8, linewidth=1.5)
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, linewidth=1.5, label='Adam→L-BFGS')
    ax.plot([], [], 'b-', linewidth=2, label=r'$\lambda_{BC/IC}$')
    ax.plot([], [], 'orange', linewidth=2, label=r'$\lambda_{data}$')
    ax.plot([], [], 'g-', linewidth=2, label=r'$\lambda_{PDE}$')
    ax.set_ylim(AXIS_LIMITS['lambdas'])
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel('Weight (log)', fontsize=FONT_CONFIG['axis_label'])
    ax.set_title('IDW Weights (Log Scale)', fontsize=FONT_CONFIG['title'])
    ax.legend(fontsize=FONT_CONFIG['legend'])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    if save_subfigures:
        path = os.path.join(subfig_dir, 'lambdas_log.pdf')
        _save_lambda_subplot(
            adam_epochs=adam_epochs, total_iterations=total_iterations,
            lam_bc=lam_bc_all, lam_data=lam_data_all, lam_f=lam_f_all,
            ylabel='Weight (log)', title='IDW Weights (Log Scale)',
            ylim=AXIS_LIMITS['lambdas'], filepath=path, use_log=True
        )
        saved_subfigures.append(path)
    
    # -------------------------------------------------------------------------
    # Plot 6: Lambda weights (LINEAR scale) - replaces duplicate log plot
    # -------------------------------------------------------------------------
    ax = axes[1, 2]
    ax.plot(range(adam_epochs), lam_bc_all[:adam_epochs], 'b-', alpha=0.8, linewidth=1.5)
    ax.plot(range(adam_epochs), lam_data_all[:adam_epochs], 'orange', alpha=0.8, linewidth=1.5)
    ax.plot(range(adam_epochs), lam_f_all[:adam_epochs], 'g-', alpha=0.8, linewidth=1.5)
    if history_lbfgs.get('lam_bc'):
        ax.plot(range(adam_epochs, total_iterations), lam_bc_all[adam_epochs:],
                'b--', alpha=0.8, linewidth=1.5)
        ax.plot(range(adam_epochs, total_iterations), lam_data_all[adam_epochs:],
                'orange', linestyle='--', alpha=0.8, linewidth=1.5)
        ax.plot(range(adam_epochs, total_iterations), lam_f_all[adam_epochs:],
                'g--', alpha=0.8, linewidth=1.5)
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, linewidth=1.5, label='Adam→L-BFGS')
    ax.plot([], [], 'b-', linewidth=2, label=r'$\lambda_{BC/IC}$')
    ax.plot([], [], 'orange', linewidth=2, label=r'$\lambda_{data}$')
    ax.plot([], [], 'g-', linewidth=2, label=r'$\lambda_{PDE}$')
    # Auto-scale y-axis for linear plot
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel('Weight', fontsize=FONT_CONFIG['axis_label'])
    ax.set_title('IDW Weights (Linear Scale)', fontsize=FONT_CONFIG['title'])
    ax.legend(fontsize=FONT_CONFIG['legend'])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    if save_subfigures:
        path = os.path.join(subfig_dir, 'lambdas_linear.pdf')
        _save_lambda_subplot(
            adam_epochs=adam_epochs, total_iterations=total_iterations,
            lam_bc=lam_bc_all, lam_data=lam_data_all, lam_f=lam_f_all,
            ylabel='Weight', title='IDW Weights (Linear Scale)',
            ylim=None, filepath=path, use_log=False
        )
        saved_subfigures.append(path)
    
    plt.tight_layout()
    
    # Save as PDF only
    filepath = os.path.join(output_dir, filename.replace('.png', '.pdf'))
    plt.savefig(filepath, dpi=DPI_SAVE, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()
    
    # Generate LaTeX snippet
    if save_subfigures and saved_subfigures:
        _append_latex_snippet(
            output_dir=output_dir,
            subfigure_paths=saved_subfigures,
            figure_type='training_diagnostics',
            caption='Training diagnostics showing D evolution (log and linear), error, losses, and IDW weights (log and linear).',
            label='fig:training_diagnostics',
            run_id=run_id
        )
    
    return {'whole': filepath, 'subfigures': saved_subfigures}


def _save_diagnostic_subplot_direct(x_data, ylabel, title, ylim, adam_epochs,
                                     d_true, d_range, d_label, filepath, use_log=True):
    """
    Save a diagnostic subplot with consistent axis limits.
    
    Args:
        x_data: List of tuples (x_vals, y_vals, linestyle, label)
        ylabel: Y-axis label
        title: Plot title
        ylim: (ymin, ymax) tuple or None for auto-scale
        adam_epochs: Iteration count for Adam phase (for vertical line)
        d_true: True D value (for horizontal line) or None
        d_range: (d_low, d_high) tuple for experimental data or None
        d_label: Label for D reference
        filepath: Output path
        use_log: Whether to use log scale on y-axis
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    for x_vals, y_vals, linestyle, label in x_data:
        if len(y_vals) > 0:
            if use_log:
                ax.semilogy(x_vals, y_vals, linestyle, alpha=0.8, linewidth=1.5, label=label)
            else:
                ax.plot(x_vals, y_vals, linestyle, alpha=0.8, linewidth=1.5, label=label)
    
    # Add reference lines
    if d_true is not None:
        ax.axhline(y=d_true, color='r', linestyle='-', linewidth=2, label=d_label)
    elif d_range is not None:
        ax.axhspan(d_range[0], d_range[1], alpha=0.3, color='green', label=d_label)
    
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, linewidth=1.5, label='Adam→L-BFGS')
    
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel(ylabel, fontsize=FONT_CONFIG['axis_label'])
    ax.set_title(title, fontsize=FONT_CONFIG['title'])
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONT_CONFIG['legend'])
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=DPI_SAVE, bbox_inches='tight')
    plt.close()


def _save_loss_subplot(adam_epochs, total_iterations, loss_bc, loss_data, loss_f,
                       ylabel, title, ylim, filepath):
    """Save loss components subplot with consistent axis limits."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    ax.semilogy(range(adam_epochs), loss_bc[:adam_epochs], 'b-', alpha=0.8, linewidth=1.5)
    ax.semilogy(range(adam_epochs), loss_data[:adam_epochs], 'orange', alpha=0.8, linewidth=1.5)
    ax.semilogy(range(adam_epochs), loss_f[:adam_epochs], 'g-', alpha=0.8, linewidth=1.5)
    
    if len(loss_bc) > adam_epochs:
        ax.semilogy(range(adam_epochs, total_iterations), loss_bc[adam_epochs:],
                    'b--', alpha=0.8, linewidth=1.5)
        ax.semilogy(range(adam_epochs, total_iterations), loss_data[adam_epochs:],
                    'orange', linestyle='--', alpha=0.8, linewidth=1.5)
        ax.semilogy(range(adam_epochs, total_iterations), loss_f[adam_epochs:],
                    'g--', alpha=0.8, linewidth=1.5)
    
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, linewidth=1.5, label='Adam→L-BFGS')
    ax.plot([], [], 'b-', linewidth=2, label=r'$\mathcal{L}_{BC/IC}$')
    ax.plot([], [], 'orange', linewidth=2, label=r'$\mathcal{L}_{data}$')
    ax.plot([], [], 'g-', linewidth=2, label=r'$\mathcal{L}_{PDE}$')
    
    ax.set_ylim(ylim)
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel(ylabel, fontsize=FONT_CONFIG['axis_label'])
    ax.set_title(title, fontsize=FONT_CONFIG['title'])
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONT_CONFIG['legend'])
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=DPI_SAVE, bbox_inches='tight')
    plt.close()


def _save_lambda_subplot(adam_epochs, total_iterations, lam_bc, lam_data, lam_f,
                         ylabel, title, ylim, filepath, use_log=True):
    """Save lambda weights subplot with consistent axis limits."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    if use_log:
        ax.semilogy(range(adam_epochs), lam_bc[:adam_epochs], 'b-', alpha=0.8, linewidth=1.5)
        ax.semilogy(range(adam_epochs), lam_data[:adam_epochs], 'orange', alpha=0.8, linewidth=1.5)
        ax.semilogy(range(adam_epochs), lam_f[:adam_epochs], 'g-', alpha=0.8, linewidth=1.5)
        
        if len(lam_bc) > adam_epochs:
            ax.semilogy(range(adam_epochs, total_iterations), lam_bc[adam_epochs:],
                        'b--', alpha=0.8, linewidth=1.5)
            ax.semilogy(range(adam_epochs, total_iterations), lam_data[adam_epochs:],
                        'orange', linestyle='--', alpha=0.8, linewidth=1.5)
            ax.semilogy(range(adam_epochs, total_iterations), lam_f[adam_epochs:],
                        'g--', alpha=0.8, linewidth=1.5)
    else:
        ax.plot(range(adam_epochs), lam_bc[:adam_epochs], 'b-', alpha=0.8, linewidth=1.5)
        ax.plot(range(adam_epochs), lam_data[:adam_epochs], 'orange', alpha=0.8, linewidth=1.5)
        ax.plot(range(adam_epochs), lam_f[:adam_epochs], 'g-', alpha=0.8, linewidth=1.5)
        
        if len(lam_bc) > adam_epochs:
            ax.plot(range(adam_epochs, total_iterations), lam_bc[adam_epochs:],
                    'b--', alpha=0.8, linewidth=1.5)
            ax.plot(range(adam_epochs, total_iterations), lam_data[adam_epochs:],
                    'orange', linestyle='--', alpha=0.8, linewidth=1.5)
            ax.plot(range(adam_epochs, total_iterations), lam_f[adam_epochs:],
                    'g--', alpha=0.8, linewidth=1.5)
    
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, linewidth=1.5, label='Adam→L-BFGS')
    ax.plot([], [], 'b-', linewidth=2, label=r'$\lambda_{BC/IC}$')
    ax.plot([], [], 'orange', linewidth=2, label=r'$\lambda_{data}$')
    ax.plot([], [], 'g-', linewidth=2, label=r'$\lambda_{PDE}$')
    
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Iteration', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel(ylabel, fontsize=FONT_CONFIG['axis_label'])
    ax.set_title(title, fontsize=FONT_CONFIG['title'])
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONT_CONFIG['legend'])
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=DPI_SAVE, bbox_inches='tight')
    plt.close()


# =============================================================================
# EXPERIMENTAL DATA VISUALIZATION
# =============================================================================

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


def plot_training_diagnostics_experimental(history_adam, history_lbfgs,
                                            output_dir='outputs', filename=None,
                                            save_subfigures=True):
    """
    Plot training diagnostics for experimental data (no ground truth D).
    
    Wrapper that calls plot_training_diagnostics with data_type='experimental'.
    """
    return plot_training_diagnostics(
        history_adam=history_adam,
        history_lbfgs=history_lbfgs,
        diff_coeff_true=None,
        data_type='experimental',
        output_dir=output_dir,
        filename=filename,
        save_subfigures=save_subfigures
    )


# =============================================================================
# LEGACY FUNCTIONS (kept for backward compatibility)
# =============================================================================

def generate_latex_snippet(subfigure_paths, caption='', label='fig:results'):
    """
    Generate LaTeX code snippet for including subfigures in a tabular layout.
    
    NOTE: This is the legacy function. LaTeX snippets are now automatically
    generated and appended to latex_snippets.txt by the main plotting functions.
    """
    filenames = [os.path.basename(p) for p in subfigure_paths]
    n_cols = min(4, len(filenames))
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    
    snippet = r"""\begin{figure}[htbp]
\centering
\setlength{\tabcolsep}{2pt}
\begin{tabular}{""" + 'c' * n_cols + r"""}
"""
    
    for row_start in range(0, len(filenames), n_cols):
        row_files = filenames[row_start:row_start + n_cols]
        row_labels = labels[row_start:row_start + len(row_files)]
        
        img_commands = [f'\\includegraphics[width={0.24}\\textwidth]{{Figs/{f}}}' 
                       for f in row_files]
        snippet += '    ' + ' &\n    '.join(img_commands) + r' \\' + '\n'
        
        label_commands = [f'\\small {lbl}' for lbl in row_labels]
        snippet += '    ' + ' & '.join(label_commands) + r' \\[0.5em]' + '\n'
    
    snippet += r"""\end{tabular}
\caption{""" + caption + r"""}
\label{""" + label + r"""}
\end{figure}
"""
    
    return snippet


def plot_2d_solution_comparison_legacy(u_pred, usol, x, y, t, diff_coeff_learned,
                                        diff_coeff_true, output_dir='outputs',
                                        filename='diff2D_IDW_inverse.png'):
    """Legacy wrapper for backward compatibility."""
    return plot_2d_solution_comparison(
        u_pred=u_pred, usol=usol, x=x, y=y, t=t,
        diff_coeff_learned=diff_coeff_learned,
        diff_coeff_true=diff_coeff_true,
        data_type='numerical',
        output_dir=output_dir,
        filename=filename,
        save_subfigures=True
    )


def plot_training_diagnostics_legacy(history_adam, history_lbfgs, diff_coeff_true,
                                      output_dir='outputs',
                                      filename='inverse_diagnostics_2D.png'):
    """Legacy wrapper for backward compatibility."""
    return plot_training_diagnostics(
        history_adam=history_adam,
        history_lbfgs=history_lbfgs,
        diff_coeff_true=diff_coeff_true,
        data_type='numerical',
        output_dir=output_dir,
        filename=filename,
        save_subfigures=True
    )

# =============================================================================
# GRADIENT HISTOGRAM DIAGNOSTICS (Maddu et al. 2021, Fig. 1)
# =============================================================================

def compute_gradient_distribution(model, loss_fn):
    """
    Compute gradient distribution for a single loss term.
    
    Args:
        model: PINN model instance
        loss_fn: Callable returning scalar loss tensor
    
    Returns:
        1D numpy array of gradient values (flattened across NN params)
    """
    import tensorflow as tf
    
    with tf.GradientTape() as tape:
        tape.watch(model.nn_variables)
        loss_val = loss_fn()
    
    grads = tape.gradient(loss_val, model.nn_variables)
    
    grad_values = []
    for g in grads:
        if g is not None:
            grad_values.extend(g.numpy().flatten())
    
    return np.array(grad_values)


def plot_gradient_histograms(
    model,
    data,
    epoch: int = None,
    weighting_method: str = None,
    output_dir: str = 'outputs',
    save_subfigures: bool = True
):
    """
    Plot gradient histograms to diagnose vanishing task-specific gradients.
    
    Replicates Fig. 1 from Maddu et al. (2021). Balanced training shows
    overlapping histograms; imbalanced training shows one dominant histogram.
    
    Args:
        model: PINN model instance
        data: Training data dict with X_u_train, u_train, X_obs, u_obs, X_f_train
        epoch: Current training epoch (for title/filename)
        weighting_method: 'IDW', 'Uniform', etc. (for title)
        output_dir: Directory to save figures
        save_subfigures: Whether to save individual subfigure
    
    Returns:
        dict: {'figure': fig, 'stats': gradient_statistics}
    """
    from ..losses.pde_losses import loss_BC, loss_Data, loss_PDE
    
    set_publication_style()
    
    # Create loss function closures
    def bc_fn():
        return loss_BC(model, data['X_u_train'], data['u_train'])
    
    def data_fn():
        return loss_Data(model, data['X_obs'], data['u_obs'])
    
    def pde_fn():
        return loss_PDE(model, data['X_f_train'])
    
    # Compute gradient distributions
    distributions = {
        r'$\nabla_\theta \mathcal{L}_{BC/IC}$': compute_gradient_distribution(model, bc_fn),
        r'$\nabla_\theta \mathcal{L}_{data}$': compute_gradient_distribution(model, data_fn),
        r'$\nabla_\theta \mathcal{L}_{PDE}$': compute_gradient_distribution(model, pde_fn),
    }
    
    # Compute statistics
    stats = {}
    for name, grads in distributions.items():
        stats[name] = {
            'variance': np.var(grads),
            'mean_abs': np.mean(np.abs(grads)),
            'max_abs': np.max(np.abs(grads)) if len(grads) > 0 else 0,
        }
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for (name, grads), color in zip(distributions.items(), colors):
        grads_filtered = grads[np.abs(grads) > 1e-12]
        if len(grads_filtered) > 0:
            ax.hist(grads_filtered, bins=50, alpha=0.6, 
                   label=name, color=color, density=True)
    
    ax.set_yscale('log')
    ax.set_xlabel('Gradient Value', fontsize=FONT_CONFIG['axis_label'])
    ax.set_ylabel('Density (log)', fontsize=FONT_CONFIG['axis_label'])
    
    # Title
    title_parts = []
    if weighting_method:
        title_parts.append(weighting_method)
    if epoch is not None:
        title_parts.append(f'Epoch {epoch}')
    ax.set_title(' - '.join(title_parts) if title_parts else 'Gradient Distributions',
                 fontsize=FONT_CONFIG['title'])
    
    ax.legend(fontsize=FONT_CONFIG['legend'])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_CONFIG['tick_label'])
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    filename = f"gradient_histograms_epoch{epoch if epoch else 0}.pdf"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=DPI_SAVE, bbox_inches='tight')
    print(f"Saved: {filepath}")
    
    # Print diagnostic
    variances = [s['variance'] for s in stats.values()]
    max_var, min_var = max(variances), min(v for v in variances if v > 0)
    var_ratio = max_var / min_var if min_var > 0 else float('inf')
    if not np.isfinite(var_ratio):
        var_ratio = 1.0  # treat as balanced for logging
        
    print(f"\nGradient Balance (epoch {epoch}):")
    print(f"  Variance ratio: {var_ratio:.1e}")
    if var_ratio > 1000:
        print(f"  ⚠️  SEVERE imbalance - vanishing task-specific gradients!")
    elif var_ratio > 100:
        print(f"  ⚠️  Moderate imbalance")
    else:
        print(f"  ✓  Reasonably balanced")
    
    return {'figure': fig, 'stats': stats, 'filepath': filepath}