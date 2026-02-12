import tensorflow as tf
import datetime, os
# hide tf logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import scipy.optimize
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import time
from pyDOE import lhs         # Latin Hypercube Sampling
import seaborn as sns
import codecs, json

# generates same random numbers each time
np.random.seed(123)
tf.random.set_seed(123)

print("TensorFlow version: {}".format(tf.__version__))

# -----------------------------
# Config (you can tweak here)
# -----------------------------
DIFF_COEFF_TRUE = 0.2         # True diffusion coefficient (for validation)
DIFF_COEFF_INIT = 0.5         # Initial guess for trainable parameter (farther from true)
IDW_EMA_BETA = 0.9            # EMA smoothing for gradient energies
IDW_EPS = 1e-12               # small epsilon to avoid divide-by-zero
IDW_CLAMP = (1e-3, 1e3)       # clamp raw weights before normalization (stability)
ADAM_LR = 1e-3
ADAM_EPOCHS = 4000            # More epochs for 2D inverse problem
PRINT_EVERY = 200
FREEZE_BEFORE_LBFGS = True    # freeze the learned weights before L-BFGS
WEIGHT_SUM_TARGET = 3.0       # three tasks now (BC/IC, Data, PDE) -> normalize to sum ~ 3

# Number of interior observation points
N_obs = 500  # Sparse observations from interior (more for 2D)

# Training parameters (will be set in main)
N_u = 200   # Total number of BC/IC data points
N_f = 2000  # Total number of collocation points (more for 2D)


# Create training data for 2D diffusion
def trainingdata_2D(inputfile, N_u, N_f, N_obs):
    """
    Load 2D diffusion data and create training sets.
    
    2D PDE: du/dt = D * (d²u/dx² + d²u/dy²)
    
    Data structure from gtDataGen2D.py:
    - usol: shape (Nt, Nx+1, Ny+1)
    - x: shape (Nx+1,)
    - y: shape (Ny+1,)
    - t: shape (Nt,)
    """
    # Read FD solution data
    data = scipy.io.loadmat(inputfile)
    x = data['x'].flatten()           # Nx+1 points
    y = data['y'].flatten()           # Ny+1 points
    t = data['t'].flatten()           # Nt time points
    usol = data['usol']               # shape (Nt, Nx+1, Ny+1)
    
    # Get true diffusion coefficient if stored
    if 'diffCoeff' in data:
        global DIFF_COEFF_TRUE
        DIFF_COEFF_TRUE = float(data['diffCoeff'])

    # Create meshgrid for all space-time points
    X, Y, T = np.meshgrid(x, y, t, indexing='ij')
    
    # Reshape usol to match meshgrid indexing
    # usol is (Nt, Nx+1, Ny+1), we need (Nx+1, Ny+1, Nt)
    usol_reshaped = np.transpose(usol, (1, 2, 0))
    
    # Create test data: all space-time points
    X_u_test = np.hstack((X.flatten()[:, None], 
                          Y.flatten()[:, None], 
                          T.flatten()[:, None]))
    u_test = usol_reshaped.flatten()[:, None]

    # Domain bounds [x_min, y_min, t_min] and [x_max, y_max, t_max]
    lb = np.array([x.min(), y.min(), t.min()])
    ub = np.array([x.max(), y.max(), t.max()])

    # -----------------------------------------
    # Boundary & Initial Conditions for 2D
    # -----------------------------------------
    all_X_u_train = []
    all_u_train = []
    
    # Initial Condition: t = 0, all x, all y
    X_ic, Y_ic = np.meshgrid(x, y, indexing='ij')
    T_ic = np.zeros_like(X_ic)
    ic_points = np.hstack((X_ic.flatten()[:, None], 
                           Y_ic.flatten()[:, None], 
                           T_ic.flatten()[:, None]))
    ic_values = usol_reshaped[:, :, 0].flatten()[:, None]
    all_X_u_train.append(ic_points)
    all_u_train.append(ic_values)
    
    # Boundary conditions at x = x_min (for all y, all t)
    Y_bc, T_bc = np.meshgrid(y, t, indexing='ij')
    X_bc = np.ones_like(Y_bc) * x.min()
    bc_x_min = np.hstack((X_bc.flatten()[:, None], 
                          Y_bc.flatten()[:, None], 
                          T_bc.flatten()[:, None]))
    bc_x_min_u = usol_reshaped[0, :, :].flatten()[:, None]
    all_X_u_train.append(bc_x_min)
    all_u_train.append(bc_x_min_u)
    
    # Boundary conditions at x = x_max
    X_bc = np.ones_like(Y_bc) * x.max()
    bc_x_max = np.hstack((X_bc.flatten()[:, None], 
                          Y_bc.flatten()[:, None], 
                          T_bc.flatten()[:, None]))
    bc_x_max_u = usol_reshaped[-1, :, :].flatten()[:, None]
    all_X_u_train.append(bc_x_max)
    all_u_train.append(bc_x_max_u)
    
    # Boundary conditions at y = y_min (for all x, all t)
    X_bc, T_bc = np.meshgrid(x, t, indexing='ij')
    Y_bc = np.ones_like(X_bc) * y.min()
    bc_y_min = np.hstack((X_bc.flatten()[:, None], 
                          Y_bc.flatten()[:, None], 
                          T_bc.flatten()[:, None]))
    bc_y_min_u = usol_reshaped[:, 0, :].flatten()[:, None]
    all_X_u_train.append(bc_y_min)
    all_u_train.append(bc_y_min_u)
    
    # Boundary conditions at y = y_max
    Y_bc = np.ones_like(X_bc) * y.max()
    bc_y_max = np.hstack((X_bc.flatten()[:, None], 
                          Y_bc.flatten()[:, None], 
                          T_bc.flatten()[:, None]))
    bc_y_max_u = usol_reshaped[:, -1, :].flatten()[:, None]
    all_X_u_train.append(bc_y_max)
    all_u_train.append(bc_y_max_u)

    # Stack all BC/IC points
    all_X_u_train = np.vstack(all_X_u_train)
    all_u_train = np.vstack(all_u_train)

    # Choose random N_u points for training BC/IC
    idx = np.random.choice(all_X_u_train.shape[0], min(N_u, all_X_u_train.shape[0]), replace=False)
    X_u_train = all_X_u_train[idx, :]
    u_train = all_u_train[idx, :]

    # -----------------------------------------
    # Collocation Points using Latin Hypercube Sampling
    # -----------------------------------------
    X_f_train = lb + (ub - lb) * lhs(3, N_f)  # 3D: (x, y, t)
    X_f_train = np.vstack((X_f_train, X_u_train))  # append training points

    # -----------------------------------------
    # Interior observation points for INVERSE PROBLEM
    # -----------------------------------------
    # Create mask for interior points (excluding boundaries and initial time)
    eps = 0.05 * (ub - lb)  # 5% margin from boundaries
    interior_mask = ((X_u_test[:, 0] > lb[0] + eps[0]) & (X_u_test[:, 0] < ub[0] - eps[0]) &
                     (X_u_test[:, 1] > lb[1] + eps[1]) & (X_u_test[:, 1] < ub[1] - eps[1]) &
                     (X_u_test[:, 2] > lb[2] + eps[2]) & (X_u_test[:, 2] < ub[2] - eps[2]))
    
    X_interior_all = X_u_test[interior_mask]
    u_interior_all = u_test[interior_mask]
    
    # Randomly select N_obs observation points
    idx_obs = np.random.choice(X_interior_all.shape[0], 
                               min(N_obs, X_interior_all.shape[0]), 
                               replace=False)
    X_obs = X_interior_all[idx_obs, :]
    u_obs = u_interior_all[idx_obs, :]

    return (X_f_train, X_u_train, u_train, X_u_test, u_test, 
            ub, lb, usol_reshaped, x, y, t, X_obs, u_obs)


# -----------------------------
# PINN with IDW loss balancing and trainable diffusion coefficient
# Now for 2D diffusion with 3 loss terms: BC/IC, Data, PDE
# -----------------------------
class Sequentialmodel(tf.Module):
    def __init__(self, layers, name=None):
        super().__init__(name=name)
        self.layers = layers

        self.W = []  # Weights and biases
        self.parameters = 0  # total number of parameters

        for i in range(len(layers) - 1):
            input_dim = layers[i]
            output_dim = layers[i + 1]
            # Xavier std
            std_dv = np.sqrt((2.0 / (input_dim + output_dim)))
            w = tf.random.normal([input_dim, output_dim], dtype='float64') * std_dv
            w = tf.Variable(w, trainable=True, name='w' + str(i + 1))
            b = tf.Variable(tf.cast(tf.zeros([output_dim]), dtype='float64'), trainable=True, name='b' + str(i + 1))
            self.W.append(w)
            self.W.append(b)
            self.parameters += input_dim * output_dim + output_dim

        # IDW trackers (EMA of squared grad norms) - 3 terms
        self.g2_bc = tf.Variable(1.0, dtype=tf.float64, trainable=False, name="g2_bc")
        self.g2_data = tf.Variable(1.0, dtype=tf.float64, trainable=False, name="g2_data")
        self.g2_f = tf.Variable(1.0, dtype=tf.float64, trainable=False, name="g2_f")
        self.beta = IDW_EMA_BETA
        self.epsw = IDW_EPS
        self.weight_sum_target = WEIGHT_SUM_TARGET

        # Whether to freeze weights for L-BFGS
        self.freeze_idw_weights = False
        self.lam_bc_fixed = tf.Variable(1.0, dtype=tf.float64, trainable=False, name="lam_bc_fixed")
        self.lam_data_fixed = tf.Variable(1.0, dtype=tf.float64, trainable=False, name="lam_data_fixed")
        self.lam_f_fixed = tf.Variable(1.0, dtype=tf.float64, trainable=False, name="lam_f_fixed")

        # Trainable diffusion coefficient
        self.diff_coeff = tf.Variable(DIFF_COEFF_INIT, dtype=tf.float64, trainable=True, name="diff_coeff")
        
        # L-BFGS iteration counter and timing
        self.lbfgs_iter = 0
        self.lbfgs_last_print_time = None

    @property
    def trainable_variables(self):
        # Include NN parameters AND the diffusion coefficient
        vars_ = []
        for i in range(len(self.layers) - 1):
            vars_.append(self.W[2 * i])
            vars_.append(self.W[2 * i + 1])
        vars_.append(self.diff_coeff)
        return vars_

    @property
    def nn_variables(self):
        # Only NN parameters (for IDW gradient computation)
        vars_ = []
        for i in range(len(self.layers) - 1):
            vars_.append(self.W[2 * i])
            vars_.append(self.W[2 * i + 1])
        return vars_

    def evaluate(self, x):
        # Normalize input using global lb/ub
        x = (x - lb) / (ub - lb)
        a = x
        for i in range(len(self.layers) - 2):
            z = tf.add(tf.matmul(a, self.W[2 * i]), self.W[2 * i + 1])
            a = tf.nn.tanh(z)
        a = tf.add(tf.matmul(a, self.W[-2]), self.W[-1])
        return a

    def get_weights(self):
        parameters_1d = []
        for i in range(len(self.layers) - 1):
            w_1d = tf.reshape(self.W[2 * i], [-1])
            b_1d = tf.reshape(self.W[2 * i + 1], [-1])
            parameters_1d = tf.concat([parameters_1d, w_1d], 0)
            parameters_1d = tf.concat([parameters_1d, b_1d], 0)
        parameters_1d = tf.concat([parameters_1d, [self.diff_coeff]], 0)
        return parameters_1d

    def set_weights(self, parameters):
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
        
        self.diff_coeff.assign(parameters[0])

    def loss_BC(self, x, y):
        """Loss for boundary/initial conditions"""
        loss_u = tf.reduce_mean(tf.square(y - self.evaluate(x)))
        return loss_u
    
    def loss_Data(self, x_obs, u_obs):
        """Loss for interior observation data - KEY FOR INVERSE PROBLEM"""
        loss_data = tf.reduce_mean(tf.square(u_obs - self.evaluate(x_obs)))
        return loss_data

    def loss_PDE(self, x_to_train_f):
        """
        2D Diffusion PDE: du/dt = D * (d²u/dx² + d²u/dy²)
        Residual: f = u_t - D * (u_xx + u_yy)
        """
        g = tf.convert_to_tensor(x_to_train_f, dtype=tf.float64)
        x_f = g[:, 0:1]
        y_f = g[:, 1:2]
        t_f = g[:, 2:3]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_f)
            tape.watch(y_f)
            tape.watch(t_f)

            xyt = tf.concat([x_f, y_f, t_f], axis=1)
            u = self.evaluate(xyt)

            # First derivatives
            u_x = tape.gradient(u, x_f)
            u_y = tape.gradient(u, y_f)
            u_t = tape.gradient(u, t_f)

        # Second derivatives
        u_xx = tape.gradient(u_x, x_f)
        u_yy = tape.gradient(u_y, y_f)

        del tape

        # Guard against None gradients
        if u_x is None:
            u_x = tf.zeros_like(x_f, dtype=tf.float64)
        if u_y is None:
            u_y = tf.zeros_like(y_f, dtype=tf.float64)
        if u_t is None:
            u_t = tf.zeros_like(t_f, dtype=tf.float64)
        if u_xx is None:
            u_xx = tf.zeros_like(x_f, dtype=tf.float64)
        if u_yy is None:
            u_yy = tf.zeros_like(y_f, dtype=tf.float64)

        # 2D diffusion residual: u_t - D*(u_xx + u_yy) = 0
        f = u_t - self.diff_coeff * (u_xx + u_yy)
        return tf.reduce_mean(tf.square(f))

    def _grad_energy(self, compute_loss_callable):
        """Compute sum of squared param-grad norms for a loss."""
        with tf.GradientTape() as tape:
            tape.watch(self.nn_variables)
            L = compute_loss_callable()
        grads = tape.gradient(L, self.nn_variables)
        g2_terms = []
        for g in grads:
            if g is not None:
                g2_terms.append(tf.reduce_sum(tf.square(g)))
        if not g2_terms:
            return tf.constant(0.0, dtype=tf.float64)
        return tf.add_n(g2_terms)

    def _compute_idw_weights(self, x_bc, y_bc, x_obs, u_obs, g):
        """Compute inverse-Dirichlet weights based on gradient energies."""
        g2_bc = self._grad_energy(lambda: self.loss_BC(x_bc, y_bc))
        g2_data = self._grad_energy(lambda: self.loss_Data(x_obs, u_obs))
        g2_f = self._grad_energy(lambda: self.loss_PDE(g))

        # EMA update
        self.g2_bc.assign(self.beta * self.g2_bc + (1.0 - self.beta) * g2_bc)
        self.g2_data.assign(self.beta * self.g2_data + (1.0 - self.beta) * g2_data)
        self.g2_f.assign(self.beta * self.g2_f + (1.0 - self.beta) * g2_f)

        # Inverse-Dirichlet raw weights
        w_bc = 1.0 / (self.g2_bc + self.epsw)
        w_data = 1.0 / (self.g2_data + self.epsw)
        w_f = 1.0 / (self.g2_f + self.epsw)

        # Clamp for stability
        w_bc = tf.clip_by_value(w_bc, IDW_CLAMP[0], IDW_CLAMP[1])
        w_data = tf.clip_by_value(w_data, IDW_CLAMP[0], IDW_CLAMP[1])
        w_f = tf.clip_by_value(w_f, IDW_CLAMP[0], IDW_CLAMP[1])

        # Normalize to fixed sum
        s = w_bc + w_data + w_f
        lam_bc = self.weight_sum_target * w_bc / s
        lam_data = self.weight_sum_target * w_data / s
        lam_f = self.weight_sum_target * w_f / s

        lam_bc = tf.stop_gradient(lam_bc)
        lam_data = tf.stop_gradient(lam_data)
        lam_f = tf.stop_gradient(lam_f)
        return lam_bc, lam_data, lam_f

    def freeze_idw(self, lam_bc_val, lam_data_val, lam_f_val):
        self.freeze_idw_weights = True
        self.lam_bc_fixed.assign(lam_bc_val)
        self.lam_data_fixed.assign(lam_data_val)
        self.lam_f_fixed.assign(lam_f_val)

    def loss(self, x_bc, y_bc, x_obs, u_obs, g):
        L_bc = self.loss_BC(x_bc, y_bc)
        L_data = self.loss_Data(x_obs, u_obs)
        L_f = self.loss_PDE(g)

        if self.freeze_idw_weights:
            lam_bc = self.lam_bc_fixed
            lam_data = self.lam_data_fixed
            lam_f = self.lam_f_fixed
        else:
            lam_bc, lam_data, lam_f = self._compute_idw_weights(x_bc, y_bc, x_obs, u_obs, g)

        loss_total = lam_bc * L_bc + lam_data * L_data + lam_f * L_f
        return loss_total, L_bc, L_data, L_f, lam_bc, lam_data, lam_f

    def optimizerfunc(self, parameters):
        """For L-BFGS optimization."""
        self.set_weights(parameters)
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            loss_val, loss_bc, loss_data, loss_f, lam_bc, lam_data, lam_f = self.loss(
                X_u_train, u_train, X_obs, u_obs, X_f_train)
        grads = tape.gradient(loss_val, self.trainable_variables)

        grads_1d = []
        for i in range(len(self.layers) - 1):
            grads_w_1d = tf.reshape(grads[2 * i], [-1])
            grads_b_1d = tf.reshape(grads[2 * i + 1], [-1])
            grads_1d = tf.concat([grads_1d, grads_w_1d], 0)
            grads_1d = tf.concat([grads_1d, grads_b_1d], 0)
        grads_1d = tf.concat([grads_1d, [grads[-1]]], 0)
        return loss_val.numpy(), grads_1d.numpy()

    def optimizer_callback(self, parameters):
        """Callback for L-BFGS that also records history."""
        global diff_coeff_history_lbfgs, loss_bc_history_lbfgs, loss_data_history_lbfgs
        global loss_f_history_lbfgs, lam_bc_history_lbfgs, lam_data_history_lbfgs, lam_f_history_lbfgs
        
        self.lbfgs_iter += 1
        
        loss_value, loss_bc, loss_data, loss_f, lam_bc, lam_data, lam_f = self.loss(
            X_u_train, u_train, X_obs, u_obs, X_f_train)
        u_pred = self.evaluate(X_u_test)
        error_vec = np.linalg.norm((u_test - u_pred), 2) / np.linalg.norm(u_test, 2)
        diff_error = np.abs(self.diff_coeff.numpy() - DIFF_COEFF_TRUE)
        
        # Record histories
        diff_coeff_history_lbfgs.append(self.diff_coeff.numpy())
        loss_bc_history_lbfgs.append(loss_bc.numpy())
        loss_data_history_lbfgs.append(loss_data.numpy())
        loss_f_history_lbfgs.append(loss_f.numpy())
        lam_bc_history_lbfgs.append(lam_bc.numpy())
        lam_data_history_lbfgs.append(lam_data.numpy())
        lam_f_history_lbfgs.append(lam_f.numpy())
        
        # Print every PRINT_EVERY iterations
        if self.lbfgs_iter % PRINT_EVERY == 0 or self.lbfgs_iter == 1:
            current_time = time.time()
            if self.lbfgs_last_print_time is not None:
                elapsed = current_time - self.lbfgs_last_print_time
                time_str = f"  dt={elapsed:.1f}s"
            else:
                time_str = ""
            self.lbfgs_last_print_time = current_time
            
            print(f"[L-BFGS {self.lbfgs_iter:5d}] L={loss_value.numpy():.3e}  Lbc={loss_bc.numpy():.3e}  "
                  f"Ldata={loss_data.numpy():.3e}  Lf={loss_f.numpy():.3e}")
            print(f"              lam_bc={lam_bc.numpy():.3e}  lam_data={lam_data.numpy():.3e}  "
                  f"lam_f={lam_f.numpy():.3e}")
            print(f"              relL2={error_vec:.3e}  D={self.diff_coeff.numpy():.4f}  D_err={diff_error:.4f}{time_str}")


def solutionplot_2D(u_pred, usol, x, y, t, diff_coeff_learned, X_obs, X_u_train):
    """Plot 2D solution comparison at different time slices (4 slices)."""
    # Reshape prediction to match solution shape
    u_pred_reshaped = np.reshape(u_pred, usol.shape, order='C')
    
    # Select 4 time indices for plotting
    nt = len(t)
    t_indices = [0, nt//3, 2*nt//3, nt-1]
    n_cols = len(t_indices)
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Use consistent colorbar limits across all time slices
    u_vmin = min(usol.min(), u_pred_reshaped.min())
    u_vmax = max(usol.max(), u_pred_reshaped.max())
    
    # Global limits for error
    error_all = np.abs(usol - u_pred_reshaped)
    err_vmin = 0
    err_vmax = error_all.max()
    
    # Create figure with GridSpec for precise control
    # 4 columns for plots + 1 narrow column for colorbar
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, n_cols + 1, width_ratios=[1, 1, 1, 1, 0.05], 
                           wspace=0.25, hspace=0.3)
    
    # Row 0: True solution
    axes_true = [fig.add_subplot(gs[0, i]) for i in range(n_cols)]
    for i, ti in enumerate(t_indices):
        im0 = axes_true[i].pcolormesh(X, Y, usol[:, :, ti], cmap='jet', shading='auto',
                                       vmin=u_vmin, vmax=u_vmax)
        axes_true[i].set_title(f'True, t={t[ti]:.3f}s', fontsize=10)
        axes_true[i].set_xlabel('x', fontsize=9)
        if i == 0:
            axes_true[i].set_ylabel('y', fontsize=9)
        axes_true[i].set_aspect('equal')
        axes_true[i].tick_params(labelsize=8)
    
    # Colorbar for row 0
    cax0 = fig.add_subplot(gs[0, n_cols])
    cbar0 = fig.colorbar(im0, cax=cax0)
    cbar0.set_label('u (True)', fontsize=9)
    cbar0.ax.tick_params(labelsize=8)
    
    # Row 1: Predicted solution
    axes_pred = [fig.add_subplot(gs[1, i]) for i in range(n_cols)]
    for i, ti in enumerate(t_indices):
        im1 = axes_pred[i].pcolormesh(X, Y, u_pred_reshaped[:, :, ti], cmap='jet', shading='auto',
                                       vmin=u_vmin, vmax=u_vmax)
        axes_pred[i].set_title(f'Predicted, t={t[ti]:.3f}s', fontsize=10)
        axes_pred[i].set_xlabel('x', fontsize=9)
        if i == 0:
            axes_pred[i].set_ylabel('y', fontsize=9)
        axes_pred[i].set_aspect('equal')
        axes_pred[i].tick_params(labelsize=8)
    
    # Colorbar for row 1
    cax1 = fig.add_subplot(gs[1, n_cols])
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.set_label('u (Pred)', fontsize=9)
    cbar1.ax.tick_params(labelsize=8)
    
    # Row 2: Error
    axes_err = [fig.add_subplot(gs[2, i]) for i in range(n_cols)]
    for i, ti in enumerate(t_indices):
        error = np.abs(usol[:, :, ti] - u_pred_reshaped[:, :, ti])
        im2 = axes_err[i].pcolormesh(X, Y, error, cmap='hot', shading='auto',
                                      vmin=err_vmin, vmax=err_vmax)
        axes_err[i].set_title(f'|Error|, t={t[ti]:.3f}s', fontsize=10)
        axes_err[i].set_xlabel('x', fontsize=9)
        if i == 0:
            axes_err[i].set_ylabel('y', fontsize=9)
        axes_err[i].set_aspect('equal')
        axes_err[i].tick_params(labelsize=8)
    
    # Colorbar for row 2
    cax2 = fig.add_subplot(gs[2, n_cols])
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.set_label('|Error|', fontsize=9)
    cbar2.ax.tick_params(labelsize=8)
    
    plt.suptitle(f'2D Diffusion: Learned D = {diff_coeff_learned:.4f} (True: {DIFF_COEFF_TRUE})', fontsize=14)
    plt.savefig('diff2D_IDW_inverse.png', dpi=300, bbox_inches='tight')
    print("Saved: diff2D_IDW_inverse.png")


# -----------------------------
# Main code
# -----------------------------
if __name__ == "__main__":
    # Training data
    inputfilename = 'gtDiff2D.mat'
    
    print("\n" + "="*60)
    print("2D Diffusion Inverse Problem with IDW Weighting")
    print("="*60)
    
    (X_f_train, X_u_train, u_train, X_u_test, u_test, 
     ub, lb, usol, x, y, t, X_obs, u_obs) = trainingdata_2D(inputfilename, N_u, N_f, N_obs)

    # Network - 3 inputs (x, y, t), deeper for 2D problem
    layers = np.array([3, 64, 64, 64, 64, 1])

    PINN = Sequentialmodel(layers)
    init_params = PINN.get_weights().numpy()

    print(f"\nStarting training...")
    print(f"  True D = {DIFF_COEFF_TRUE}, Initial D = {DIFF_COEFF_INIT}")

    # -----------------------------
    # Adam warm-up with dynamic IDW
    # -----------------------------
    optimizer = tf.keras.optimizers.Adam(learning_rate=ADAM_LR)

    # Track histories for Adam phase
    diff_coeff_history = []
    loss_bc_history = []
    loss_data_history = []
    loss_f_history = []
    lam_bc_history = []
    lam_data_history = []
    lam_f_history = []

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            loss_val, loss_bc, loss_data, loss_f, lam_bc, lam_data, lam_f = PINN.loss(
                X_u_train, u_train, X_obs, u_obs, X_f_train)
        grads = tape.gradient(loss_val, PINN.trainable_variables)
        optimizer.apply_gradients(zip(grads, PINN.trainable_variables))
        return loss_val, loss_bc, loss_data, loss_f, lam_bc, lam_data, lam_f

    print("\n--- Adam warm-up (IDW dynamic) ---")
    adam_start_time = time.time()
    last_print_time = adam_start_time
    
    for epoch in range(1, ADAM_EPOCHS + 1):
        loss_val, L_bc, L_data, L_f, lam_bc, lam_data, lam_f = train_step()
        
        # Record histories
        diff_coeff_history.append(PINN.diff_coeff.numpy())
        loss_bc_history.append(L_bc.numpy())
        loss_data_history.append(L_data.numpy())
        loss_f_history.append(L_f.numpy())
        lam_bc_history.append(lam_bc.numpy())
        lam_data_history.append(lam_data.numpy())
        lam_f_history.append(lam_f.numpy())
        
        if epoch % PRINT_EVERY == 0 or epoch == 1 or epoch == ADAM_EPOCHS:
            current_time = time.time()
            elapsed = current_time - last_print_time
            last_print_time = current_time
            
            u_pred_tmp = PINN.evaluate(X_u_test)
            err_tmp = np.linalg.norm((u_test - u_pred_tmp), 2) / np.linalg.norm(u_test, 2)
            diff_err = np.abs(PINN.diff_coeff.numpy() - DIFF_COEFF_TRUE)
            print(f"[Adam {epoch:5d}] L={loss_val.numpy():.3e}  Lbc={L_bc.numpy():.3e}  "
                  f"Ldata={L_data.numpy():.3e}  Lf={L_f.numpy():.3e}")
            print(f"             lam_bc={lam_bc.numpy():.3e}  lam_data={lam_data.numpy():.3e}  "
                  f"lam_f={lam_f.numpy():.3e}")
            print(f"             relL2={err_tmp:.3e}  D={PINN.diff_coeff.numpy():.4f}  D_err={diff_err:.4f}  dt={elapsed:.1f}s")

    adam_total_time = time.time() - adam_start_time

    # Freeze IDW weights before L-BFGS
    if FREEZE_BEFORE_LBFGS:
        _, _, _, _, lam_bc_last, lam_data_last, lam_f_last = PINN.loss(
            X_u_train, u_train, X_obs, u_obs, X_f_train)
        PINN.freeze_idw(lam_bc_last.numpy(), lam_data_last.numpy(), lam_f_last.numpy())
        print(f"\nFroze IDW weights: lam_bc={lam_bc_last.numpy():.6f}, "
              f"lam_data={lam_data_last.numpy():.6f}, lam_f={lam_f_last.numpy():.6f}")

    # -----------------------------
    # L-BFGS-B optimization
    # -----------------------------
    print("\n--- L-BFGS fine-tuning ---")
    start_time = time.time()

    # Initialize L-BFGS histories
    diff_coeff_history_lbfgs = []
    loss_bc_history_lbfgs = []
    loss_data_history_lbfgs = []
    loss_f_history_lbfgs = []
    lam_bc_history_lbfgs = []
    lam_data_history_lbfgs = []
    lam_f_history_lbfgs = []
    
    # Reset L-BFGS iteration counter
    PINN.lbfgs_iter = 0
    PINN.lbfgs_last_print_time = time.time()

    results = scipy.optimize.minimize(fun=PINN.optimizerfunc,
                                      x0=PINN.get_weights().numpy(),
                                      args=(),
                                      method='L-BFGS-B',
                                      jac=True,
                                      callback=PINN.optimizer_callback,
                                      options={'disp': None,
                                               'maxcor': 200,
                                               'ftol': 1 * np.finfo(float).eps,
                                               'gtol': 5e-8,
                                               'maxfun': 2000,
                                               'maxiter': 4000,
                                               'iprint': -1,
                                               'maxls': 50})

    elapsed = time.time() - start_time

    PINN.set_weights(results.x)

    # -----------------------------
    # Model Accuracy
    # -----------------------------
    u_pred = PINN.evaluate(X_u_test)
    error_vec = np.linalg.norm((u_test - u_pred), 2) / np.linalg.norm(u_test, 2)

    diff_coeff_learned = PINN.diff_coeff.numpy()
    diff_coeff_error = np.abs(diff_coeff_learned - DIFF_COEFF_TRUE)

    # Solution plot
    solutionplot_2D(u_pred.numpy(), usol, x, y, t, diff_coeff_learned, X_obs, X_u_train)

    # -----------------------------
    # Diagnostic plots (extended to include L-BFGS phase)
    # -----------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Combine histories from both phases
    diff_coeff_all = diff_coeff_history + diff_coeff_history_lbfgs
    loss_bc_all = loss_bc_history + loss_bc_history_lbfgs
    loss_data_all = loss_data_history + loss_data_history_lbfgs
    loss_f_all = loss_f_history + loss_f_history_lbfgs
    lam_bc_all = lam_bc_history + lam_bc_history_lbfgs
    lam_data_all = lam_data_history + lam_data_history_lbfgs
    lam_f_all = lam_f_history + lam_f_history_lbfgs
    
    adam_epochs = len(diff_coeff_history)
    total_iterations = len(diff_coeff_all)

    # Plot 1: D evolution (linear)
    ax = axes[0, 0]
    ax.plot(range(adam_epochs), diff_coeff_history, 'b-', label='Adam phase')
    if diff_coeff_history_lbfgs:
        ax.plot(range(adam_epochs, total_iterations), diff_coeff_history_lbfgs, 'g-', label='L-BFGS phase')
    ax.axhline(y=DIFF_COEFF_TRUE, color='r', linestyle='--', label=f'True D = {DIFF_COEFF_TRUE}')
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Diffusion Coefficient')
    ax.set_title('D Evolution (Linear Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: D evolution (log)
    ax = axes[0, 1]
    ax.semilogy(range(adam_epochs), diff_coeff_history, 'b-', label='Adam phase')
    if diff_coeff_history_lbfgs:
        ax.semilogy(range(adam_epochs, total_iterations), diff_coeff_history_lbfgs, 'g-', label='L-BFGS phase')
    ax.axhline(y=DIFF_COEFF_TRUE, color='r', linestyle='--', label=f'True D = {DIFF_COEFF_TRUE}')
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Diffusion Coefficient (log)')
    ax.set_title('D Evolution (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: D error (log)
    ax = axes[0, 2]
    d_errors_adam = np.abs(np.array(diff_coeff_history) - DIFF_COEFF_TRUE)
    ax.semilogy(range(adam_epochs), d_errors_adam, 'b-', label='Adam phase')
    if diff_coeff_history_lbfgs:
        d_errors_lbfgs = np.abs(np.array(diff_coeff_history_lbfgs) - DIFF_COEFF_TRUE)
        ax.semilogy(range(adam_epochs, total_iterations), d_errors_lbfgs, 'g-', label='L-BFGS phase')
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('|D - D_true| (log scale)')
    ax.set_title('Diffusion Coefficient Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Loss components (now includes L-BFGS phase)
    ax = axes[1, 0]
    ax.semilogy(range(adam_epochs), loss_bc_history, 'b-', alpha=0.8)
    ax.semilogy(range(adam_epochs), loss_data_history, 'orange', alpha=0.8)
    ax.semilogy(range(adam_epochs), loss_f_history, 'g-', alpha=0.8)
    if loss_bc_history_lbfgs:
        ax.semilogy(range(adam_epochs, total_iterations), loss_bc_history_lbfgs, 'b--', alpha=0.8)
        ax.semilogy(range(adam_epochs, total_iterations), loss_data_history_lbfgs, 'orange', linestyle='--', alpha=0.8)
        ax.semilogy(range(adam_epochs, total_iterations), loss_f_history_lbfgs, 'g--', alpha=0.8)
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, label='Adam→L-BFGS')
    ax.plot([], [], 'b-', label='L_bc (BC/IC)')
    ax.plot([], [], 'orange', label='L_data (Interior)')
    ax.plot([], [], 'g-', label='L_f (PDE)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components (solid=Adam, dashed=L-BFGS)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Lambda weights (linear, now includes L-BFGS phase)
    ax = axes[1, 1]
    ax.plot(range(adam_epochs), lam_bc_history, 'b-', alpha=0.8)
    ax.plot(range(adam_epochs), lam_data_history, 'orange', alpha=0.8)
    ax.plot(range(adam_epochs), lam_f_history, 'g-', alpha=0.8)
    if lam_bc_history_lbfgs:
        ax.plot(range(adam_epochs, total_iterations), lam_bc_history_lbfgs, 'b--', alpha=0.8)
        ax.plot(range(adam_epochs, total_iterations), lam_data_history_lbfgs, 'orange', linestyle='--', alpha=0.8)
        ax.plot(range(adam_epochs, total_iterations), lam_f_history_lbfgs, 'g--', alpha=0.8)
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, label='Adam→L-BFGS')
    ax.plot([], [], 'b-', label='λ_bc')
    ax.plot([], [], 'orange', label='λ_data')
    ax.plot([], [], 'g-', label='λ_f')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Weight')
    ax.set_title('IDW Weights (solid=Adam, dashed=L-BFGS)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Lambda weights (log, now includes L-BFGS phase)
    ax = axes[1, 2]
    ax.semilogy(range(adam_epochs), lam_bc_history, 'b-', alpha=0.8)
    ax.semilogy(range(adam_epochs), lam_data_history, 'orange', alpha=0.8)
    ax.semilogy(range(adam_epochs), lam_f_history, 'g-', alpha=0.8)
    if lam_bc_history_lbfgs:
        ax.semilogy(range(adam_epochs, total_iterations), lam_bc_history_lbfgs, 'b--', alpha=0.8)
        ax.semilogy(range(adam_epochs, total_iterations), lam_data_history_lbfgs, 'orange', linestyle='--', alpha=0.8)
        ax.semilogy(range(adam_epochs, total_iterations), lam_f_history_lbfgs, 'g--', alpha=0.8)
    ax.axvline(x=adam_epochs, color='k', linestyle=':', alpha=0.5, label='Adam→L-BFGS')
    ax.plot([], [], 'b-', label='λ_bc')
    ax.plot([], [], 'orange', label='λ_data')
    ax.plot([], [], 'g-', label='λ_f')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Weight (log)')
    ax.set_title('IDW Weights Log (solid=Adam, dashed=L-BFGS)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('inverse_diagnostics_2D.png', dpi=300)
    print("Saved: inverse_diagnostics_2D.png")
    plt.show()

    # -----------------------------
    # Final Comprehensive Summary
    # -----------------------------
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print("\n--- Configuration ---")
    print(f"  DIFF_COEFF_TRUE     = {DIFF_COEFF_TRUE}")
    print(f"  DIFF_COEFF_INIT     = {DIFF_COEFF_INIT}")
    print(f"  IDW_EMA_BETA        = {IDW_EMA_BETA}")
    print(f"  IDW_EPS             = {IDW_EPS}")
    print(f"  IDW_CLAMP           = {IDW_CLAMP}")
    print(f"  WEIGHT_SUM_TARGET   = {WEIGHT_SUM_TARGET}")
    print(f"  FREEZE_BEFORE_LBFGS = {FREEZE_BEFORE_LBFGS}")
    print(f"  ADAM_LR             = {ADAM_LR}")
    print(f"  ADAM_EPOCHS         = {ADAM_EPOCHS}")
    
    print("\n--- Data ---")
    print(f"  Input file          = {inputfilename}")
    print(f"  nx={len(x)}, ny={len(y)}, nt={len(t)}")
    print(f"  Domain: x in [{x.min():.2f},{x.max():.2f}], y in [{y.min():.2f},{y.max():.2f}], t in [{t.min():.2f},{t.max():.2f}]")
    print(f"  N_u (BC/IC points)  = {N_u}")
    print(f"  N_f (collocation)   = {N_f}")
    print(f"  N_obs (interior)    = {N_obs}")
    
    print("\n--- Network ---")
    print(f"  Layers              = {list(layers)}")
    print(f"  Total parameters    = {PINN.parameters + 1}")
    
    print("\n--- Training Time ---")
    print(f"  Adam time           = {adam_total_time:.2f}s")
    print(f"  L-BFGS time         = {elapsed:.2f}s")
    print(f"  Total time          = {adam_total_time + elapsed:.2f}s")
    print(f"  L-BFGS iterations   = {results.nit}")
    print(f"  L-BFGS func evals   = {results.nfev}")
    lbfgs_msg = results.message.decode() if isinstance(results.message, bytes) else results.message
    print(f"  L-BFGS termination  = {lbfgs_msg}")
    
    print("\n--- Results ---")
    print(f"  D_true              = {DIFF_COEFF_TRUE}")
    print(f"  D_learned           = {diff_coeff_learned:.6f}")
    print(f"  D_error             = {diff_coeff_error:.6f} ({100*diff_coeff_error/DIFF_COEFF_TRUE:.2f}%)")
    print(f"  Relative L2 error   = {error_vec:.5e}")
    
    print("\n--- Final IDW Weights ---")
    final_lam_bc = lam_bc_history_lbfgs[-1] if lam_bc_history_lbfgs else lam_bc_history[-1]
    final_lam_data = lam_data_history_lbfgs[-1] if lam_data_history_lbfgs else lam_data_history[-1]
    final_lam_f = lam_f_history_lbfgs[-1] if lam_f_history_lbfgs else lam_f_history[-1]
    print(f"  lambda_bc           = {final_lam_bc:.6f}")
    print(f"  lambda_data         = {final_lam_data:.6f}")
    print(f"  lambda_f            = {final_lam_f:.6f}")
    
    print("\n--- Output Files ---")
    print(f"  diff2D_IDW_inverse.png")
    print(f"  inverse_diagnostics_2D.png")
    
    print("="*70)