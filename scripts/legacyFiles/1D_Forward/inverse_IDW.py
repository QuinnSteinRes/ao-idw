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
DIFF_COEFF_INIT = 0.5         # Initial guess for trainable parameter
IDW_EMA_BETA = 0.9            # EMA smoothing for gradient energies
IDW_EPS = 1e-12               # small epsilon to avoid divide-by-zero
IDW_CLAMP = (1e-3, 1e3)       # clamp raw weights before normalization (stability)
ADAM_LR = 1e-3
ADAM_EPOCHS = 5000            # Adam warm-up epochs before L-BFGS
PRINT_EVERY = 200
FREEZE_BEFORE_LBFGS = True    # freeze the learned weights before L-BFGS
WEIGHT_SUM_TARGET = 2.0       # two tasks (BC/IC and PDE) -> normalize to sum ~ 2

# Create training data
def trainingdata(inputfile, N_u, N_f):
    # Read FD solution data
    data = scipy.io.loadmat(inputfile)  # Load data from file
    x = data['x'].T                     # nx points between -1 and 1 [nxX1]
    t = data['t'].T                     # nt time points between 0 and 1 [ntX1]
    usol = data['usol'].T               # solution of nxXnt grid points
    print("nx =", x.shape, "nt =", t.shape, "usol(nx,nt) =", usol.shape)

    X, T = np.meshgrid(x, t)            # u(X[i],T[j]) = usol[i][j]

    # Create test data
    X_u_test = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    # Domain bounds
    lb = X_u_test[0]   # [-1. 0.]
    ub = X_u_test[-1]  # [1.  0.99]
    u = usol.flatten('F')[:, None]  # array nxXnt

    # Boundary & Initial Conditions
    # Initial Condition: -1 <= x <= 1 at t = 0
    leftedge_x = np.hstack((X[0, :][:, None], T[0, :][:, None]))  # L1
    leftedge_u = usol[:, 0][:, None]

    # Boundary x = -1 and 0 <= t <= 1
    bottomedge_x = np.hstack((X[:, 0][:, None], T[:, 0][:, None]))  # L2
    bottomedge_u = usol[-1, :][:, None]

    # Boundary x = 1 and 0 <= t <= 1
    topedge_x = np.hstack((X[:, -1][:, None], T[:, 0][:, None]))    # L3
    topedge_u = usol[0, :][:, None]

    all_X_u_train = np.vstack([leftedge_x, bottomedge_x, topedge_x])
    all_u_train = np.vstack([leftedge_u, bottomedge_u, topedge_u])

    # choose random N_u points for training
    idx = np.random.choice(all_X_u_train.shape[0], N_u, replace=False)
    X_u_train = all_X_u_train[idx, :]
    u_train = all_u_train[idx, :]

    # Collocation Points
    # Latin Hypercube sampling for collocation points
    X_f_train = lb + (ub - lb) * lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))  # append training points to collocation points

    return X_f_train, X_u_train, u_train, X_u_test, u, ub, lb, usol, x, t

# -----------------------------
# PINN with IDW loss balancing and trainable diffusion coefficient
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

        # IDW trackers (EMA of squared grad norms)
        self.g2_u = tf.Variable(1.0, dtype=tf.float64, trainable=False, name="g2_u")
        self.g2_f = tf.Variable(1.0, dtype=tf.float64, trainable=False, name="g2_f")
        self.beta = IDW_EMA_BETA
        self.epsw = IDW_EPS
        self.weight_sum_target = WEIGHT_SUM_TARGET

        # Whether to freeze weights for L-BFGS
        self.freeze_idw_weights = False
        self.lam_u_fixed = tf.Variable(1.0, dtype=tf.float64, trainable=False, name="lam_u_fixed")
        self.lam_f_fixed = tf.Variable(1.0, dtype=tf.float64, trainable=False, name="lam_f_fixed")

        # Trainable diffusion coefficient (initialized to initial guess)
        self.diff_coeff = tf.Variable(DIFF_COEFF_INIT, dtype=tf.float64, trainable=True, name="diff_coeff")

    @property
    def trainable_variables(self):
        # Include NN parameters AND the diffusion coefficient
        vars_ = []
        for i in range(len(self.layers) - 1):
            vars_.append(self.W[2 * i])
            vars_.append(self.W[2 * i + 1])
        vars_.append(self.diff_coeff)  # Add physics parameter
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
        # use global lb/ub; they will be defined after data loading
        x = (x - lb) / (ub - lb)
        a = x
        for i in range(len(self.layers) - 2):
            z = tf.add(tf.matmul(a, self.W[2 * i]), self.W[2 * i + 1])
            a = tf.nn.tanh(z)
        a = tf.add(tf.matmul(a, self.W[-2]), self.W[-1])  # no activation on output
        return a

    def get_weights(self):
        parameters_1d = []
        for i in range(len(self.layers) - 1):
            w_1d = tf.reshape(self.W[2 * i], [-1])
            b_1d = tf.reshape(self.W[2 * i + 1], [-1])
            parameters_1d = tf.concat([parameters_1d, w_1d], 0)
            parameters_1d = tf.concat([parameters_1d, b_1d], 0)
        # Append diffusion coefficient
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
        
        # Set diffusion coefficient (last parameter)
        self.diff_coeff.assign(parameters[0])

    def loss_BC(self, x, y):
        loss_u = tf.reduce_mean(tf.square(y - self.evaluate(x)))
        return loss_u

    def loss_PDE(self, x_to_train_f):
        # Convert once; no Variables inside tf.function
        g = tf.convert_to_tensor(x_to_train_f, dtype=tf.float64)
        x_f = g[:, 0:1]
        t_f = g[:, 1:2]

        # Single persistent tape: watch x_f and t_f, then compute u, u_x, u_t, and u_xx
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_f)
            tape.watch(t_f)

            # IMPORTANT: use concat (not stack) so gradients wrt x_f are preserved cleanly
            xt = tf.concat([x_f, t_f], axis=1)   # shape (N, 2)
            u = self.evaluate(xt)               # u(x,t)

            # first derivatives (computed while tape is active)
            u_x = tape.gradient(u, x_f)
            u_t = tape.gradient(u, t_f)

        # second derivative (still using the SAME persistent tape)
        u_xx = tape.gradient(u_x, x_f)

        # clean up persistent resources
        del tape

        # Guard against None (can happen early if graph wasn't connected); fall back to zeros to keep training going
        if u_x is None:
            u_x = tf.zeros_like(x_f, dtype=tf.float64)
        if u_t is None:
            u_t = tf.zeros_like(t_f, dtype=tf.float64)
        if u_xx is None:
            u_xx = tf.zeros_like(x_f, dtype=tf.float64)

        # Use trainable diffusion coefficient
        f = u_t - self.diff_coeff * u_xx
        return tf.reduce_mean(tf.square(f))
    

    def _grad_energy(self, compute_loss_callable):
        '''Re-evaluate the single loss inside a tape and return sum of squared param-grad norms.'''
        with tf.GradientTape() as tape:
            tape.watch(self.nn_variables)  # Use nn_variables for IDW (not physics params)
            L = compute_loss_callable()
        grads = tape.gradient(L, self.nn_variables)
        g2_terms = []
        for g in grads:
            if g is not None:
                g2_terms.append(tf.reduce_sum(tf.square(g)))
        if not g2_terms:
            return tf.constant(0.0, dtype=tf.float64)
        return tf.add_n(g2_terms)

    def _compute_idw_weights(self, x, y, g):
        # Compute gradient energies with respect to NN parameters (not physics params)
        g2_u = self._grad_energy(lambda: self.loss_BC(x, y))
        g2_f = self._grad_energy(lambda: self.loss_PDE(g))

        # EMA update
        self.g2_u.assign(self.beta * self.g2_u + (1.0 - self.beta) * g2_u)
        self.g2_f.assign(self.beta * self.g2_f + (1.0 - self.beta) * g2_f)

        # inverse-Dirichlet raw weights
        w_u = 1.0 / (self.g2_u + self.epsw)
        w_f = 1.0 / (self.g2_f + self.epsw)

        # clamp for stability
        w_u = tf.clip_by_value(w_u, IDW_CLAMP[0], IDW_CLAMP[1])
        w_f = tf.clip_by_value(w_f, IDW_CLAMP[0], IDW_CLAMP[1])

        # normalize to fixed sum
        s = w_u + w_f
        lam_u = self.weight_sum_target * w_u / s
        lam_f = self.weight_sum_target * w_f / s

        # Stop gradient through the weighting (pure reweighting heuristic)
        lam_u = tf.stop_gradient(lam_u)
        lam_f = tf.stop_gradient(lam_f)
        return lam_u, lam_f

    def freeze_idw(self, lam_u_val, lam_f_val):
        self.freeze_idw_weights = True
        self.lam_u_fixed.assign(lam_u_val)
        self.lam_f_fixed.assign(lam_f_val)

    def loss(self, x, y, g):
        # Compute task-specific losses
        L_u = self.loss_BC(x, y)
        L_f = self.loss_PDE(g)

        if self.freeze_idw_weights:
            lam_u = self.lam_u_fixed
            lam_f = self.lam_f_fixed
        else:
            lam_u, lam_f = self._compute_idw_weights(x, y, g)

        loss_total = lam_u * L_u + lam_f * L_f
        return loss_total, L_u, L_f, lam_u, lam_f

    def optimizerfunc(self, parameters):
        # For L-BFGS (after freezing weights ideally)
        self.set_weights(parameters)
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            loss_val, loss_u, loss_f, lam_u, lam_f = self.loss(X_u_train, u_train, X_f_train)
        grads = tape.gradient(loss_val, self.trainable_variables)

        grads_1d = []
        for i in range(len(self.layers) - 1):
            grads_w_1d = tf.reshape(grads[2 * i], [-1])
            grads_b_1d = tf.reshape(grads[2 * i + 1], [-1])
            grads_1d = tf.concat([grads_1d, grads_w_1d], 0)
            grads_1d = tf.concat([grads_1d, grads_b_1d], 0)
        # Append gradient for diffusion coefficient
        grads_1d = tf.concat([grads_1d, [grads[-1]]], 0)
        return loss_val.numpy(), grads_1d.numpy()

    def optimizer_callback(self, parameters):
        loss_value, loss_u, loss_f, lam_u, lam_f = self.loss(X_u_train, u_train, X_f_train)
        u_pred = self.evaluate(X_u_test)
        error_vec = np.linalg.norm((u - u_pred), 2) / np.linalg.norm(u, 2)
        diff_error = np.abs(self.diff_coeff.numpy() - DIFF_COEFF_TRUE)
        tf.print("L:", loss_value, "Lu:", loss_u, "Lf:", loss_f,
                 "lam_u:", lam_u, "lam_f:", lam_f,
                 "relL2:", error_vec, "D:", self.diff_coeff, "D_err:", diff_error)

# Plot comparison
def solutionplot(u_pred, X_u_train, X_f_train, u_train, usol, x, t, diff_coeff_learned):
    fig, ax = plt.subplots()
    ax.axis('off')

    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(u_pred, interpolation='nearest', cmap='rainbow',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx',
            label='BC & IC Data (%d points)' % (u_train.shape[0]),
            markersize=4, clip_on=False)
    ax.plot(X_f_train[:, 1], X_f_train[:, 0], 'ro',
            label='PDE Data (%d points)' % (X_f_train[:, 1].shape[0] - u_train.shape[0]),
            markersize=1, clip_on=False)

    line = np.linspace(x.min(), x.max(), 2)[:, None]

    # Time instants
    t1 = int(1)
    t2 = int(200)
    t3 = int(375)

    ax.plot(t[t1]*np.ones((2,1)), line, 'w-', linewidth=1)
    ax.plot(t[t2]*np.ones((2,1)), line, 'w-', linewidth=1)
    ax.plot(t[t3]*np.ones((2,1)), line, 'w-', linewidth=1)

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, ncol=2)
    ax.set_title(f'$u(x,t)$ | Learned $D$ = {diff_coeff_learned:.4f} (True: {DIFF_COEFF_TRUE})', fontsize=10)

    ####### Row 1: u(t,x) slices ##################
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, usol.T[t1, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, u_pred.T[t1, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.set_title('$t = 0s$', fontsize=10)
    ax.axis('square')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, usol.T[t2, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, u_pred.T[t2, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.axis('square')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title('$t = 0.50s$', fontsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x, usol.T[t3, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, u_pred.T[t3, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.axis('square')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title('$t = 0.75s$', fontsize=10)

    plt.savefig('diff1D_IDW_inverse.png', dpi=500)

# -----------------------------
# Main code
# -----------------------------
# Training and testing
N_u = 25   # Total number of data points for 'u'
N_f = 100  # Total number of collocation points

# Training data
inputfilename = 'gtDiff1D.mat'
X_f_train, X_u_train, u_train, X_u_test, u, ub, lb, usol, x, t = trainingdata(inputfilename, N_u, N_f)

# Network
# layers = np.array([2,20,20,20,20,20,20,20,20,1])   # deeper net
layers = np.array([2, 20, 1])  # shallow net as in your original

PINN = Sequentialmodel(layers)
init_params = PINN.get_weights().numpy()

print(f"\nTrue diffusion coefficient: {DIFF_COEFF_TRUE}")
print(f"Initial guess: {DIFF_COEFF_INIT}")

# -----------------------------
# Adam warm-up with dynamic IDW
# -----------------------------
optimizer = tf.keras.optimizers.Adam(learning_rate=ADAM_LR)

# Track diffusion coefficient history
diff_coeff_history = []

@tf.function
def train_step():
    with tf.GradientTape() as tape:
        loss_val, loss_u, loss_f, lam_u, lam_f = PINN.loss(X_u_train, u_train, X_f_train)
    grads = tape.gradient(loss_val, PINN.trainable_variables)
    optimizer.apply_gradients(zip(grads, PINN.trainable_variables))
    return loss_val, loss_u, loss_f, lam_u, lam_f

print("\n--- Adam warm-up (IDW dynamic) ---")
for epoch in range(1, ADAM_EPOCHS + 1):
    loss_val, L_u, L_f, lam_u, lam_f = train_step()
    diff_coeff_history.append(PINN.diff_coeff.numpy())
    
    if epoch % PRINT_EVERY == 0 or epoch == 1 or epoch == ADAM_EPOCHS:
        u_pred_tmp = PINN.evaluate(X_u_test)
        err_tmp = np.linalg.norm((u - u_pred_tmp), 2) / np.linalg.norm(u, 2)
        diff_err = np.abs(PINN.diff_coeff.numpy() - DIFF_COEFF_TRUE)
        print(f"[Adam {epoch:5d}] L={loss_val.numpy():.3e}  Lu={L_u.numpy():.3e}  Lf={L_f.numpy():.3e}  "
              f"lam_u={lam_u.numpy():.3e}  lam_f={lam_f.numpy():.3e}  relL2={err_tmp:.3e}  "
              f"D={PINN.diff_coeff.numpy():.4f}  D_err={diff_err:.4f}")

# Optionally freeze IDW weights before L-BFGS
if FREEZE_BEFORE_LBFGS:
    # capture latest weights
    _, _, _, lam_u_last, lam_f_last = PINN.loss(X_u_train, u_train, X_f_train)
    PINN.freeze_idw(lam_u_last.numpy(), lam_f_last.numpy())
    print(f"Froze IDW weights before L-BFGS: lam_u={lam_u_last.numpy():.6f}, lam_f={lam_f_last.numpy():.6f}")

# -----------------------------
# L-BFGS-B optimization (fine-tune)
# -----------------------------
print("\n--- L-BFGS fine-tuning ---")
start_time = time.time()

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
                                           'maxfun': 5000,
                                           'maxiter': 500,
                                           'iprint': -1,
                                           'maxls': 50})

elapsed = time.time() - start_time
print('Training time: %.2f' % (elapsed))
print(results)

PINN.set_weights(results.x)

# -----------------------------
# Model Accuracy & Plot
# -----------------------------
u_pred = PINN.evaluate(X_u_test)
error_vec = np.linalg.norm((u - u_pred), 2) / np.linalg.norm(u, 2)
print('Test Error: %.5f' % (error_vec))

diff_coeff_learned = PINN.diff_coeff.numpy()
diff_coeff_error = np.abs(diff_coeff_learned - DIFF_COEFF_TRUE)
print(f'\nDiffusion coefficient:')
print(f'  True:    {DIFF_COEFF_TRUE}')
print(f'  Learned: {diff_coeff_learned:.6f}')
print(f'  Error:   {diff_coeff_error:.6f} ({100*diff_coeff_error/DIFF_COEFF_TRUE:.2f}%)')

u_pred = np.reshape(u_pred, usol.shape, order='F')  # Fortran style
solutionplot(u_pred, X_u_train, X_f_train, u_train, usol, x, t, diff_coeff_learned)

# Plot diffusion coefficient evolution
plt.figure(figsize=(10, 4))
plt.plot(diff_coeff_history, 'b-', label='Learned D')
plt.axhline(y=DIFF_COEFF_TRUE, color='r', linestyle='--', label=f'True D = {DIFF_COEFF_TRUE}')
plt.xlabel('Adam Epoch')
plt.ylabel('Diffusion Coefficient')
plt.title('Evolution of Learned Diffusion Coefficient During Training')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('diff_coeff_evolution.png', dpi=300)
plt.show()