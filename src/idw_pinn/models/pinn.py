"""
PINN model for 2D diffusion inverse problems.
Extracted from legacy: Sequentialmodel class
"""
import tensorflow as tf
import numpy as np


class PINN(tf.Module):
    """
    Physics-Informed Neural Network for 2D diffusion.
    
    Supports:
    - Forward pass with input normalization
    - Trainable diffusion coefficient (inverse problem)
    - IDW weighting state tracking (for training module)
    - L-BFGS optimizer compatibility via get/set_weights
    
    NOTE FOR MIGRATION: The following attributes/methods are trainer concerns
    and should be moved to trainer.py when implementing:
    - lbfgs_iter, lbfgs_last_print_time
    - optimizer_callback() method
    These are kept here temporarily to match legacy structure.
    """
    
    def __init__(
        self,
        layers,
        lb,
        ub,
        diff_coeff_init,
        idw_config,
        intensity_scaling_enabled: bool = False,
        intensity_scale_init: float = 1.0,
        intensity_bias_init: float = 0.0,
        name=None,
    ):
        """
        Initialize PINN model.
        
        Args:
            layers: List of layer sizes [input_dim, hidden1, ..., output_dim]
            lb: Lower bounds for input normalization (numpy array)
            ub: Upper bounds for input normalization (numpy array)
            diff_coeff_init: Initial value for trainable diffusion coefficient
            idw_config: Config object with IDW parameters (ema_beta, eps, etc.)
            name: Optional name for the module
        """
        super().__init__(name=name)
        self.layers = layers
        self.lb = tf.constant(lb, dtype=tf.float64)
        self.ub = tf.constant(ub, dtype=tf.float64)
        
        # Initialize network weights and biases
        self.W = []
        self.parameters = 0
        
        for i in range(len(layers) - 1):
            input_dim = layers[i]
            output_dim = layers[i + 1]
            
            # Xavier initialization
            std_dv = np.sqrt((2.0 / (input_dim + output_dim)))
            w = tf.random.normal([input_dim, output_dim], dtype='float64') * std_dv
            w = tf.Variable(w, trainable=True, name=f'w{i+1}')
            b = tf.Variable(tf.cast(tf.zeros([output_dim]), dtype='float64'), 
                          trainable=True, name=f'b{i+1}')
            
            self.W.append(w)
            self.W.append(b)
            self.parameters += input_dim * output_dim + output_dim
        
        # Trainable diffusion coefficient (inverse problem): enforce positivity
        # Parameterization: D = softplus(D_raw) + D_min
        self.diff_coeff_min = tf.constant(1e-16, dtype=tf.float64)
        # Initialize raw so that softplus(raw) ~= diff_coeff_init (clipped to >0)
        d0 = float(diff_coeff_init) if diff_coeff_init is not None else 1.0
        d0 = max(d0, 1e-12)
        d_raw0 = np.log(np.expm1(d0))  # inverse softplus
        self.diff_coeff_raw = tf.Variable(d_raw0, dtype=tf.float64, trainable=True, name='diff_coeff_raw')
        # Optional learnable mapping from network output ("concentration") to observed intensity
        #   I_pred(x,y,t) = a * u_pred(x,y,t) + b
        # where a is constrained positive for identifiability/stability.
        self.intensity_scaling_enabled = bool(intensity_scaling_enabled)
        if self.intensity_scaling_enabled:
            # Parameterize a via softplus for positivity.
            # Initialize a_raw so that softplus(a_raw) ~= intensity_scale_init.
            a0 = float(intensity_scale_init)
            # inverse softplus approx: a_raw = log(exp(a0) - 1)
            a_raw0 = np.log(np.exp(a0) - 1.0) if a0 > 1e-12 else -30.0
            self.intensity_scale_raw = tf.Variable(a_raw0, dtype=tf.float64, trainable=True, name="intensity_scale_raw")
            self.intensity_bias = tf.Variable(float(intensity_bias_init), dtype=tf.float64, trainable=True, name="intensity_bias")
        
        # IDW weighting state (used by training module)
        self.g2_bc = tf.Variable(1.0, dtype=tf.float64, trainable=False, name="g2_bc")
        self.g2_data = tf.Variable(1.0, dtype=tf.float64, trainable=False, name="g2_data")
        self.g2_f = tf.Variable(1.0, dtype=tf.float64, trainable=False, name="g2_f")
        
        self.beta = idw_config.idw.ema_beta
        self.epsw = idw_config.idw.eps
        self.weight_sum_target = idw_config.idw.weight_sum_target
        
        # IDW freezing for L-BFGS
        self.freeze_idw_weights = False
        self.lam_bc_fixed = tf.Variable(1.0, dtype=tf.float64, trainable=False, name="lam_bc_fixed")
        self.lam_data_fixed = tf.Variable(1.0, dtype=tf.float64, trainable=False, name="lam_data_fixed")
        self.lam_f_fixed = tf.Variable(1.0, dtype=tf.float64, trainable=False, name="lam_f_fixed")
        
        # L-BFGS tracking (NOTE: Move to trainer.py during migration)
        self.lbfgs_iter = 0
        self.lbfgs_last_print_time = None
    
    @property
    def trainable_variables(self):
        """Return all trainable variables (NN params + diff_coeff)."""
        vars_ = []
        for i in range(len(self.layers) - 1):
            vars_.append(self.W[2 * i])      # weights
            vars_.append(self.W[2 * i + 1])  # biases
        vars_.append(self.diff_coeff_raw)
        if getattr(self, "intensity_scaling_enabled", False):
            vars_.append(self.intensity_scale_raw)
            vars_.append(self.intensity_bias)
        return vars_
    
    @property
    def nn_variables(self):
        """Return only NN parameters (for IDW gradient computation)."""
        vars_ = []
        for i in range(len(self.layers) - 1):
            vars_.append(self.W[2 * i])      # weights
            vars_.append(self.W[2 * i + 1])  # biases
        return vars_
    
    def evaluate(self, x):
        """
        Forward pass through network.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Output tensor (batch_size, output_dim)
        """
        # Normalize input to [0, 1]
        x = (x - self.lb) / (self.ub - self.lb)
        
        # Forward pass with tanh activation
        a = x
        for i in range(len(self.layers) - 2):
            z = tf.add(tf.matmul(a, self.W[2 * i]), self.W[2 * i + 1])
            a = tf.nn.tanh(z)
        
        # Output layer (no activation)
        a = tf.add(tf.matmul(a, self.W[-2]), self.W[-1])
        return a

    def intensity_scale(self):
        """Positive intensity scale a (softplus parameterization)."""
        if not getattr(self, "intensity_scaling_enabled", False):
            return tf.constant(1.0, dtype=tf.float64)
        return tf.nn.softplus(self.intensity_scale_raw)

    def predict_observation(self, x):
        """Prediction in the observation space (intensity)."""
        u = self.evaluate(x)
        if not getattr(self, "intensity_scaling_enabled", False):
            return u
        return self.intensity_scale() * u + self.intensity_bias
    
    def evaluate_intensity(self, X):
        """Convenience wrapper to evaluate prediction in observation (intensity) space.

        Returns:
            If intensity scaling enabled: I = a*u + b
            Else: returns u
        """
        # predict_observation expects same input type as evaluate()
        return self.predict_observation(X)
        
    def get_weights(self):
        """
        Get flattened parameter vector (for L-BFGS optimizer).
        
        Returns:
            1D tensor with all parameters [W1, b1, W2, b2, ..., diff_coeff]
        """
        parts = []
        for i in range(len(self.layers) - 1):
            w_1d = tf.reshape(self.W[2 * i], [-1])
            b_1d = tf.reshape(self.W[2 * i + 1], [-1])
            parts.append(w_1d)
            parts.append(b_1d)
        parts.append(tf.reshape(self.diff_coeff_raw, [-1]))
        if getattr(self, "intensity_scaling_enabled", False):
            parts.append(tf.reshape(self.intensity_scale_raw, [-1]))
            parts.append(tf.reshape(self.intensity_bias, [-1]))
        return tf.concat(parts, axis=0)
    
    def set_weights(self, parameters):
        """
        Set parameters from flattened vector (for L-BFGS optimizer).
        
        Args:
            parameters: 1D array with all parameters
        """
        parameters = np.array(parameters)
        for i in range(len(self.layers) - 1):
            shape_w = tf.shape(self.W[2 * i]).numpy()
            size_w = tf.size(self.W[2 * i]).numpy()
            shape_b = tf.shape(self.W[2 * i + 1]).numpy()
            size_b = tf.size(self.W[2 * i + 1]).numpy()
            
            pick_w = parameters[0:size_w]
            self.W[2 * i].assign(tf.reshape(pick_w, shape_w))
            parameters = np.delete(parameters, np.arange(size_w), 0)
            
            pick_b = parameters[0:size_b]
            self.W[2 * i + 1].assign(tf.reshape(pick_b, shape_b))
            parameters = np.delete(parameters, np.arange(size_b), 0)
        
        self.diff_coeff_raw.assign(parameters[0])
        parameters = np.delete(parameters, 0, 0)

        if getattr(self, "intensity_scaling_enabled", False):
            self.intensity_scale_raw.assign(parameters[0])
            self.intensity_bias.assign(parameters[1])
    
    def freeze_idw(self, lam_bc_val, lam_data_val, lam_f_val):
        """
        Freeze IDW weights at current values (for L-BFGS phase).
        
        Args:
            lam_bc_val: BC/IC loss weight
            lam_data_val: Data loss weight
            lam_f_val: PDE residual loss weight
        """
        self.freeze_idw_weights = True
        self.lam_bc_fixed.assign(lam_bc_val)
        self.lam_data_fixed.assign(lam_data_val)
        self.lam_f_fixed.assign(lam_f_val)
    def diff_coeff_value(self):
        return tf.nn.softplus(self.diff_coeff_raw) + self.diff_coeff_min


    @property
    def diff_coeff(self):
        """Backwards-compatible access to positive diffusion coefficient."""
        return self.diff_coeff_value()