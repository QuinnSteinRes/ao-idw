"""
Inverse-Dirichlet Weighting (IDW) computation for PINN training.
Extracted from legacy: 2D_num_inv_IDW_newPrintOut_newFigs.py

Core algorithm from:
"Inverse-Dirichlet Weighting Enables Reliable Training of Physics Informed Neural Networks"
Maddu et al. (2021)

Key principle: Weight losses inversely proportional to their gradient variance (Dirichlet energy)
to prevent vanishing task-specific gradients in multi-objective optimization.
"""
import tensorflow as tf


def compute_grad_energy(model, loss_fn):
    """
    Compute Dirichlet energy (sum of squared gradients) for a single loss term.
    
    This measures the "gradient magnitude" of a loss with respect to network parameters.
    Used to balance losses in multi-objective optimization - losses with larger gradient
    energies dominate training, so we weight them inversely.
    
    Args:
        model: PINN model instance with nn_variables property
        loss_fn: Callable that returns a scalar loss tensor
        
    Returns:
        Scalar tensor: sum of squared gradients over all NN parameters
        
    Note:
        Only computes gradients w.r.t. NN parameters (model.nn_variables),
        NOT physics parameters like diff_coeff. This preserves the theoretical
        foundation of IDW which balances task-specific gradients.
    """
    with tf.GradientTape() as tape:
        tape.watch(model.nn_variables)
        loss_val = loss_fn()
    
    grads = tape.gradient(loss_val, model.nn_variables)
    
    # Sum squared gradients over all parameters
    g2_terms = []
    for g in grads:
        if g is not None:
            g2_terms.append(tf.reduce_sum(tf.square(g)))
    
    if not g2_terms:
        return tf.constant(0.0, dtype=tf.float64)
    
    return tf.add_n(g2_terms)


def update_idw_weights(model, x_bc, y_bc, x_obs, u_obs, x_f, loss_fns, config):
    """
    Compute and update Inverse-Dirichlet weights for multi-objective PINN training.
    
    This implements the core IDW algorithm:
    1. Compute gradient energy (Dirichlet energy) for each loss term
    2. Update exponential moving average (EMA) of gradient energies
    3. Compute inverse weights: w_k = 1 / (g2_k + eps)
    4. Clip weights for numerical stability
    5. Normalize weights to sum to weight_sum_target (typically 3.0 for 3 tasks)
    
    Args:
        model: PINN model instance with IDW state variables
        x_bc, y_bc: Boundary/initial condition data
        x_obs, u_obs: Interior observation data
        x_f: Collocation points for PDE residual
        loss_fns: Dict with keys 'bc', 'data', 'pde' containing loss callables
        config: Config object with IDW parameters
        
    Returns:
        Tuple of (lam_bc, lam_data, lam_f): Normalized weight tensors with stop_gradient
        
    Updates:
        model.g2_bc, model.g2_data, model.g2_f: EMA trackers for gradient energies
    """
    # Compute gradient energies for each loss component
    g2_bc = compute_grad_energy(model, loss_fns['bc'])
    g2_data = compute_grad_energy(model, loss_fns['data'])
    g2_f = compute_grad_energy(model, loss_fns['pde'])
    
    # EMA update of gradient energy trackers
    beta = model.beta
    model.g2_bc.assign(beta * model.g2_bc + (1.0 - beta) * g2_bc)
    model.g2_data.assign(beta * model.g2_data + (1.0 - beta) * g2_data)
    model.g2_f.assign(beta * model.g2_f + (1.0 - beta) * g2_f)
    
    # Inverse-Dirichlet raw weights: 1 / (g2 + eps)
    # Larger gradient energies → smaller weights (inverse relationship)
    epsw = model.epsw
    w_bc = 1.0 / (model.g2_bc + epsw)
    w_data = 1.0 / (model.g2_data + epsw)
    w_f = 1.0 / (model.g2_f + epsw)
    
    # Clamp for numerical stability
    clamp_min = config.idw.clamp_min
    clamp_max = config.idw.clamp_max
    w_bc = tf.clip_by_value(w_bc, clamp_min, clamp_max)
    w_data = tf.clip_by_value(w_data, clamp_min, clamp_max)
    w_f = tf.clip_by_value(w_f, clamp_min, clamp_max)
    
    # Normalize to fixed sum (typically 3.0 for 3 tasks)
    # This ensures weights are comparable across different problem scales
    s = w_bc + w_data + w_f
    target = model.weight_sum_target
    lam_bc = target * w_bc / s
    lam_data = target * w_data / s
    lam_f = target * w_f / s
    
    # Stop gradient to prevent weights from being optimized
    # Weights are hyperparameters, not learnable parameters
    lam_bc = tf.stop_gradient(lam_bc)
    lam_data = tf.stop_gradient(lam_data)
    lam_f = tf.stop_gradient(lam_f)
    
    return lam_bc, lam_data, lam_f