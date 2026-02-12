"""
IDW-PINN: Inverse-Dirichlet Weighted Physics-Informed Neural Networks

Implementation of Physics-Informed Neural Networks with Inverse-Dirichlet Weighting
for solving inverse problems in diffusion equations.

Main components:
- Config: YAML-based configuration management
- PINN: Neural network model with trainable physics parameters
- IDWPINNTrainer: Two-phase training (Adam + L-BFGS) with IDW loss balancing
- load_2d_diffusion_data: Data loading for 2D diffusion problems
- Visualization utilities for solution comparison and training diagnostics

Example usage:
    >>> from idw_pinn import Config, PINN, IDWPINNTrainer, load_2d_diffusion_data
    >>> 
    >>> config = Config('configs/default_2d_inverse.yaml')
    >>> data = load_2d_diffusion_data(config)
    >>> model = PINN(
    ...     layers=config.network.layers,
    ...     lb=data['bounds'][0],
    ...     ub=data['bounds'][1],
    ...     diff_coeff_init=config.physics.diff_coeff_init,
    ...     idw_config=config
    ... )
    >>> trainer = IDWPINNTrainer(model, config, data)
    >>> results = trainer.train()

Reference:
    Maddu et al. (2021) "Inverse-Dirichlet Weighting Enables Reliable Training
    of Physics Informed Neural Networks"
"""

from .config import Config
from .models import PINN
from .training import IDWPINNTrainer
from .data import load_2d_diffusion_data
from .utils import plot_2d_solution_comparison, plot_training_diagnostics

__version__ = '0.1.0'
__author__ = 'Quinn'

__all__ = [
    'Config',
    'PINN',
    'IDWPINNTrainer',
    'load_2d_diffusion_data',
    'plot_2d_solution_comparison',
    'plot_training_diagnostics',
]