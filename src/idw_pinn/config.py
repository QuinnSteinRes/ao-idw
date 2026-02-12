"""
Configuration management for IDW-PINN training.

Provides structured access to all hyperparameters via YAML configuration files.
Replaces hardcoded constants from legacy implementation with centralized config.

Configuration sections:
    physics: True/initial diffusion coefficients for inverse problems
    idw: Inverse-Dirichlet Weighting parameters
        - ema_beta: EMA smoothing for gradient energies (typically 0.9)
        - eps: Small epsilon to avoid divide-by-zero (typically 1e-12)
        - clamp_min/max: Stability bounds for raw weights (1e-3, 1e3)
        - weight_sum_target: Normalization target (3.0 for three loss terms)
        - freeze_before_lbfgs: Whether to freeze weights before L-BFGS phase
    training: Optimizer settings for Adam warm-up and L-BFGS fine-tuning
    data: Input file paths and sampling parameters (BC/IC, collocation, observations)
    network: Layer architecture specification

Example:
    >>> config = Config('configs/default_2d_inverse.yaml')
    >>> print(config.physics.diff_coeff_true)
    0.2
    >>> print(config.idw.ema_beta)
    0.9
    >>> print(config.network.layers)
    [3, 64, 64, 64, 64, 1]

The Config object is passed to data loaders, models, and trainers to ensure
consistent hyperparameter access throughout the pipeline.
"""
import yaml
from typing import Any


class Config:
    """
    Configuration container with nested attribute access.
    
    Loads YAML files and converts nested dictionaries into objects
    with dot notation access (e.g., config.physics.diff_coeff_true).
    
    Args:
        config_path: Path to YAML configuration file
    """
    
    def __init__(self, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        for key, value in config_dict.items():
            setattr(self, key, self._dict_to_obj(value))
    
    def _dict_to_obj(self, d: Any) -> Any:
        """
        Recursively convert nested dicts to objects with attribute access.
        
        Args:
            d: Dictionary or other value to convert
            
        Returns:
            NestedConfig object if d is dict, otherwise returns d unchanged
        """
        if isinstance(d, dict):
            return NestedConfig(d)
        return d


class NestedConfig:
    """
    Nested configuration object for hierarchical parameter access.
    
    Enables dot notation for nested configuration sections:
        config.idw.ema_beta instead of config['idw']['ema_beta']
    """
    
    def __init__(self, d: dict):
        """Convert dictionary to object with attribute access."""
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, NestedConfig(value))
            else:
                setattr(self, key, value)