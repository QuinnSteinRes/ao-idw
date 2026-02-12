"""Data loading utilities for IDW-PINN."""

from .loader import load_2d_diffusion_data
from .csv_loader import load_csv_diffusion_data, convert_diffusion_coefficient

__all__ = [
    'load_2d_diffusion_data',
    'load_csv_diffusion_data', 
    'convert_diffusion_coefficient'
]


def load_data(config):
    """
    Load data based on format specified in config.
    
    Automatically selects the appropriate loader based on:
    1. config.data.data_format if specified ('csv' or 'mat')
    2. File extension of config.data.input_file
    
    Args:
        config: Configuration object
        
    Returns:
        dict: Training data, test data, bounds, and metadata
    """
    # Check if format explicitly specified
    data_format = getattr(config.data, 'data_format', None)
    
    if data_format is None:
        # Infer from file extension
        input_file = config.data.input_file
        if input_file.endswith('.csv'):
            data_format = 'csv'
        elif input_file.endswith('.mat'):
            data_format = 'mat'
        else:
            raise ValueError(f"Cannot infer data format from: {input_file}")
    
    if data_format == 'csv':
        return load_csv_diffusion_data(config)
    elif data_format == 'mat':
        return load_2d_diffusion_data(config)
    else:
        raise ValueError(f"Unknown data format: {data_format}")
