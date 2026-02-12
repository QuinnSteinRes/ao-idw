"""
Training diagnostic plots for monitoring PINN convergence.

Visualizes diffusion coefficient evolution, errors, loss components, and
IDW lambda weights throughout training. Supports both Adam and L-BFGS phases.

Updated: January 2026
"""
import numpy as np
import matplotlib.pyplot as plt
import os

from .styles import (set_publication_style, FONT_CONFIG, DPI_SAVE,
                     AXIS_LIMITS, DATA_CONFIG, DATA_TYPE)
from .utils import generate_unique_filename, ensure_output_dirs, append_latex_snippet


def _save_diagnostic_subplot_direct(x_data, ylabel, title, ylim, adam_epochs,
                                     d_true, d_range, d_label, filepath, use_log=True):
    """Save a diagnostic subplot with consistent axis limits."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    for x_vals, y_vals, linestyle, label in x_data:
        if len(y_vals) > 0:
            if use_log:
                ax.semilogy(x_vals, y_vals, linestyle, alpha=0.8, linewidth=1.5, label=label)
            else:
                ax.plot(x_vals, y_vals, linestyle, alpha=0.8, linewidth=1.5, label=label)
    
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
    if data_type is None:
        data_type = DATA_TYPE
        
    set_publication_style()
    subfig_dir, run_id = ensure_output_dirs(output_dir)
    
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
    
    # Plot 1: D evolution (LOG scale)
    ax = axes[0, 0]
    ax.semilogy(range(adam_epochs), history_adam['diff_coeff'], 'b-', alpha=0.8, linewidth=1.5, label='Adam')
    if history_lbfgs.get('diff_coeff'):
        ax.semilogy(range(adam_epochs, total_iterations), history_lbfgs['diff_coeff'],
                'b--', alpha=0.8, linewidth=1.5, label='L-BFGS')
    
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
    
    # Plot 2: D evolution (LINEAR scale)
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
    
    # Plot 3: D error evolution
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
    
    # Plot 4: Loss components (LOG scale)
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
    
    # Plot 5: Lambda weights (LOG scale)
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
    
    # Plot 6: Lambda weights (LINEAR scale)
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
        append_latex_snippet(
            output_dir=output_dir,
            subfigure_paths=saved_subfigures,
            figure_type='training_diagnostics',
            caption='Training diagnostics showing D evolution (log and linear), error, losses, and IDW weights (log and linear).',
            label='fig:training_diagnostics',
            run_id=run_id
        )
    
    return {'whole': filepath, 'subfigures': saved_subfigures}


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