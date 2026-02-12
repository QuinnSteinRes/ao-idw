"""Loss functions for 2D diffusion inverse problems."""

from .pde_losses import loss, loss_BC, loss_Data, loss_PDE

__all__ = ['loss', 'loss_BC', 'loss_Data', 'loss_PDE']