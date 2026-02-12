"""
Plotting utilities for IDW-PINN visualization.

Public API for creating publication-ready plots of solution comparisons,
training diagnostics, and gradient distributions.

Updated: January 2026
"""

from .solution_plots import plot_2d_solution_comparison, plot_experimental_comparison
from .diagnostics import plot_training_diagnostics, plot_training_diagnostics_experimental
from .gradients import plot_gradient_histograms, compute_gradient_distribution
from .styles import set_publication_style, DATA_TYPE
from .utils import reset_session_run_id

__all__ = [
    # Solution visualization
    'plot_2d_solution_comparison',
    'plot_experimental_comparison',
    
    # Training diagnostics
    'plot_training_diagnostics',
    'plot_training_diagnostics_experimental',
    
    # Gradient analysis
    'plot_gradient_histograms',
    'compute_gradient_distribution',
    
    # Configuration
    'set_publication_style',
    'DATA_TYPE',
    'reset_session_run_id',
]