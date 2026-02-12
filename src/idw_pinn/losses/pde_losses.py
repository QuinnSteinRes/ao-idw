"""
PDE loss functions for 2D diffusion inverse problems.
Extracted from legacy: 2D_num_inv_IDW_newPrintOut_newFigs.py

Contains:
- loss_BC: Boundary/initial condition loss
- loss_Data: Interior observation data loss  
- loss_PDE: 2D diffusion PDE residual
- loss: Combined weighted loss with IDW support
"""
import tensorflow as tf


def loss_BC(model, x_bc, y_bc):
    """
    Loss for boundary/initial conditions.
    
    Args:
        model: PINN model instance
        x_bc: BC/IC input points (N_bc, 3)
        y_bc: BC/IC target values (N_bc, 1)
        
    Returns:
        Scalar loss tensor
    """
    # If BC/IC is not used (empty arrays), return zero.
    if x_bc is None or y_bc is None:
        return tf.constant(0.0, dtype=tf.float64)
    if tf.size(x_bc) == 0:
        return tf.constant(0.0, dtype=tf.float64)
    # Boundary/initial conditions usually correspond to the physical field u
    # (not the observation/intensity mapping). We therefore use model.evaluate.
    loss_u = tf.reduce_mean(tf.square(y_bc - model.evaluate(x_bc)))
    return loss_u


def loss_Data(model, x_obs, u_obs):
    """
    Loss for interior observation data - KEY FOR INVERSE PROBLEM.
    
    Args:
        model: PINN model instance
        x_obs: Observation input points (N_obs, 3)
        u_obs: Observation target values (N_obs, 1)
        
    Returns:
        Scalar loss tensor
    """
    # Interior observations may be in *intensity* space; if the model has
    # intensity scaling enabled, map NN output to observation space.
    pred = model.predict_observation(x_obs) if hasattr(model, "predict_observation") else model.evaluate(x_obs)
    loss_data = tf.reduce_mean(tf.square(u_obs - pred))
    return loss_data


def loss_PDE(model, x_f):
    """
    2D Diffusion PDE residual loss.
    
    PDE: du/dt = D * (d²u/dx² + d²u/dy²)
    Residual: f = u_t - D * (u_xx + u_yy)
    
    Args:
        model: PINN model instance
        x_f: Collocation points (N_f, 3) where columns are [x, y, t]
        
    Returns:
        Scalar loss tensor (mean squared residual)
    """
    g = tf.convert_to_tensor(x_f, dtype=tf.float64)
    x_f_col = g[:, 0:1]
    y_f_col = g[:, 1:2]
    t_f_col = g[:, 2:3]
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_f_col)
        tape.watch(y_f_col)
        tape.watch(t_f_col)
        
        xyt = tf.concat([x_f_col, y_f_col, t_f_col], axis=1)
        u = model.evaluate(xyt)
        
        # First derivatives
        u_x = tape.gradient(u, x_f_col)
        u_y = tape.gradient(u, y_f_col)
        u_t = tape.gradient(u, t_f_col)
    
    # Second derivatives
    u_xx = tape.gradient(u_x, x_f_col)
    u_yy = tape.gradient(u_y, y_f_col)
    
    del tape
    
    # Guard against None gradients
    if u_x is None:
        u_x = tf.zeros_like(x_f_col, dtype=tf.float64)
    if u_y is None:
        u_y = tf.zeros_like(y_f_col, dtype=tf.float64)
    if u_t is None:
        u_t = tf.zeros_like(t_f_col, dtype=tf.float64)
    if u_xx is None:
        u_xx = tf.zeros_like(x_f_col, dtype=tf.float64)
    if u_yy is None:
        u_yy = tf.zeros_like(y_f_col, dtype=tf.float64)
    
    # 2D diffusion residual: u_t - D*(u_xx + u_yy) = 0
    f = u_t - model.diff_coeff * (u_xx + u_yy)
    return tf.reduce_mean(tf.square(f))


def _grad_energy(model, compute_loss_callable):
    """
    Compute sum of squared param-grad norms for a loss (Dirichlet energy).
    
    Used for IDW weight computation. Only computes gradients w.r.t. NN parameters,
    not physics parameters like diff_coeff.
    
    Args:
        model: PINN model instance
        compute_loss_callable: Function that returns loss scalar
        
    Returns:
        Scalar tensor: sum of squared gradients over NN parameters
    """
    with tf.GradientTape() as tape:
        tape.watch(model.nn_variables)
        L = compute_loss_callable()
    grads = tape.gradient(L, model.nn_variables)
    g2_terms = []
    for g in grads:
        if g is not None:
            g2_terms.append(tf.reduce_sum(tf.square(g)))
    if not g2_terms:
        return tf.constant(0.0, dtype=tf.float64)
    return tf.add_n(g2_terms)


def _compute_idw_weights(model, x_bc, y_bc, x_obs, u_obs, x_f, config):
    """
    Compute inverse-Dirichlet weights based on gradient energies (IDW).

    Key behavior for experimental pointcloud runs:
    - If there are NO BC/IC samples (n_u == 0), we *drop* the BC/IC term from
      the weighting (lam_bc = 0) and normalize only over (data, PDE).
    - The target sum of lambdas is chosen automatically as the number of
      *active* loss components (2 or 3), so you do not need to specify it
      in the YAML.

    Returns:
        (lam_bc, lam_data, lam_f) as float64 tensors.
    """
    # Uniform weights mode
    if hasattr(config.idw, 'enabled') and not config.idw.enabled:
        one = tf.constant(1.0, dtype=tf.float64)
        return one, one, one

    # Determine which components are active from sample counts
    def _has_samples(x):
            if x is None:
                return False
            # Works for numpy arrays and tensors with known leading dimension
            n0 = getattr(x, 'shape', [None])[0]
            if n0 is None:
                return False
            return int(n0) > 0

    has_bc = _has_samples(x_bc) and _has_samples(y_bc)
    has_data = _has_samples(x_obs) and _has_samples(u_obs)
    has_f = _has_samples(x_f)

    # Compute gradient energies ONLY for active components
    if has_bc:
        g2_bc = _grad_energy(model, lambda: loss_BC(model, x_bc, y_bc))
        beta = model.beta
        model.g2_bc.assign(beta * model.g2_bc + (1.0 - beta) * g2_bc)

    if has_data:
        g2_data = _grad_energy(model, lambda: loss_Data(model, x_obs, u_obs))
        beta = model.beta
        model.g2_data.assign(beta * model.g2_data + (1.0 - beta) * g2_data)

    if has_f:
        g2_f = _grad_energy(model, lambda: loss_PDE(model, x_f))
        beta = model.beta
        model.g2_f.assign(beta * model.g2_f + (1.0 - beta) * g2_f)

    # Inverse-Dirichlet raw weights: 1 / (g2 + eps)
    epsw = model.epsw
    w_bc = tf.constant(0.0, dtype=tf.float64)
    w_data = tf.constant(0.0, dtype=tf.float64)
    w_f = tf.constant(0.0, dtype=tf.float64)

    if has_bc:
        w_bc = 1.0 / (model.g2_bc + epsw)
    if has_data:
        w_data = 1.0 / (model.g2_data + epsw)
    if has_f:
        w_f = 1.0 / (model.g2_f + epsw)

    # Clamp for stability (only affects active weights)
    clamp_min = config.idw.clamp_min
    clamp_max = config.idw.clamp_max
    w_bc = tf.clip_by_value(w_bc, clamp_min, clamp_max)
    w_data = tf.clip_by_value(w_data, clamp_min, clamp_max)
    w_f = tf.clip_by_value(w_f, clamp_min, clamp_max)

    # Drop BC weighting entirely if no BC samples
    if not has_bc:
        w_bc = tf.constant(0.0, dtype=tf.float64)

    # Normalize over active components
    s = w_bc + w_data + w_f + tf.constant(1e-300, dtype=tf.float64)

    # Target sum chosen automatically (#active terms: 2 or 3)
    n_active = int(has_bc) + int(has_data) + int(has_f)
    if n_active <= 0:
        n_active = 1
    target = tf.constant(float(n_active), dtype=tf.float64)

    lam_bc = target * w_bc / s
    lam_data = target * w_data / s
    lam_f = target * w_f / s

    # If BC is inactive, guarantee lam_bc==0 exactly
    if not has_bc:
        lam_bc = tf.constant(0.0, dtype=tf.float64)

    # Stop gradient so weights are not optimized
    lam_bc = tf.stop_gradient(lam_bc)
    lam_data = tf.stop_gradient(lam_data)
    lam_f = tf.stop_gradient(lam_f)

    return lam_bc, lam_data, lam_f


def loss(model, x_bc, y_bc, x_obs, u_obs, x_f, config):
    """
    Combined weighted loss for 2D diffusion inverse problem.
    
    Computes three loss components (BC/IC, Data, PDE) and combines them
    with IDW weights (dynamic or frozen).
    
    Args:
        model: PINN model instance
        x_bc, y_bc: BC/IC data
        x_obs, u_obs: Interior observation data
        x_f: Collocation points for PDE residual
        config: Config object
        
    Returns:
        Tuple: (loss_total, L_bc, L_data, L_f, lam_bc, lam_data, lam_f)
            - loss_total: Weighted sum of losses
            - L_bc, L_data, L_f: Individual loss components
            - lam_bc, lam_data, lam_f: Applied weights
    """
    # Compute individual loss components
    L_bc = loss_BC(model, x_bc, y_bc)
    L_data = loss_Data(model, x_obs, u_obs)
    L_f = loss_PDE(model, x_f)
    
    # Get weights (frozen or dynamic)
    if model.freeze_idw_weights:
        lam_bc = model.lam_bc_fixed
        lam_data = model.lam_data_fixed
        lam_f = model.lam_f_fixed
    else:
        lam_bc, lam_data, lam_f = _compute_idw_weights(
            model, x_bc, y_bc, x_obs, u_obs, x_f, config
        )
    
    # Weighted sum
    loss_total = lam_bc * L_bc + lam_data * L_data + lam_f * L_f
    
    return loss_total, L_bc, L_data, L_f, lam_bc, lam_data, lam_f