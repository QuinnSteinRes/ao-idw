"""
Gradient histogram diagnostics for monitoring training balance.

Implements gradient distribution visualization as described in Maddu et al. (2021)
Figure 1. Balanced training shows overlapping histograms; imbalanced training 
shows one dominant histogram indicating vanishing task-specific gradients.

Updated: January 2026
"""
import numpy as np
import matplotlib.pyplot as plt
import os

from .styles import set_publication_style, FONT_CONFIG, DPI_SAVE


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
        dict: {'figure': fig, 'stats': gradient_statistics, 'filepath': path}
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
        print(f"  SEVERE imbalance - vanishing task-specific gradients!")
    elif var_ratio > 100:
        print(f"  Moderate imbalance")
    else:
        print(f"  OK, reasonably balanced")
    
    return {'figure': fig, 'stats': stats, 'filepath': filepath}