"""Utilities for IDW-PINN."""

from .visualization import (
    plot_2d_solution_comparison, 
    plot_training_diagnostics,
    plot_gradient_histograms  # Add this
)

__all__ = [
    'plot_2d_solution_comparison', 
    'plot_training_diagnostics',
    'plot_gradient_histograms'  # Add this
]