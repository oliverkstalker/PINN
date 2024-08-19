import numpy as np
import tensorflow as tf


# Define the number of points
N_f = 10000  # Number of collocation points
N_IC = 200  # Number of initial condition points
N_BC = 200  # Number of boundary condition points

# Define the spatial and time domain
x_min, x_max = 0.0, 1.0
t_min, t_max = 0.0, 1.0

# Generate collocation points (x, t) within the domain
x_f = np.random.uniform(x_min, x_max, N_f)
t_f = np.random.uniform(t_min, t_max, N_f)

# Generate initial condition points (x, t=0)
x_IC = np.random.uniform(x_min, x_max, N_IC)
t_IC = np.zeros(N_IC)

# Generate boundary condition points (t) at x=0 and x=1
t_BC = np.random.uniform(t_min, t_max, N_BC)
x_BC_0 = np.full(N_BC, x_min)
x_BC_1 = np.full(N_BC, x_max)

# Convert to TensorFlow tensors
x_f = tf.convert_to_tensor(x_f, dtype=tf.float32)
t_f = tf.convert_to_tensor(t_f, dtype=tf.float32)
x_IC = tf.convert_to_tensor(x_IC, dtype=tf.float32)
t_IC = tf.convert_to_tensor(t_IC, dtype=tf.float32)
t_BC = tf.convert_to_tensor(t_BC, dtype=tf.float32)
x_BC_0 = tf.convert_to_tensor(x_BC_0, dtype=tf.float32)
x_BC_1 = tf.convert_to_tensor(x_BC_1, dtype=tf.float32)

# Reshape tensors to have two columns (x and t)
x_f = tf.reshape(x_f, (-1, 1))
t_f = tf.reshape(t_f, (-1, 1))
x_IC = tf.reshape(x_IC, (-1, 1))
t_IC = tf.reshape(t_IC, (-1, 1))
t_BC = tf.reshape(t_BC, (-1, 1))
x_BC_0 = tf.reshape(x_BC_0, (-1, 1))
x_BC_1 = tf.reshape(x_BC_1, (-1, 1))

# Combine x and t to create the input tensors
X_f = np.hstack([x_f, t_f])
X_IC = np.hstack([x_IC, t_IC])
X_BC_0 = np.hstack([x_BC_0, t_BC])
X_BC_1 = np.hstack([x_BC_1, t_BC])

# Normalize the inputs to the range [-1, 1]
X_f = (X_f - 0.5) * 2
X_IC = (X_IC - 0.5) * 2
X_BC_0 = (X_BC_0 - 0.5) * 2
X_BC_1 = (X_BC_1 - 0.5) * 2

# Print shapes of the tensors to verify
print("Collocation points (X_f):", X_f.shape)
print("Initial condition points (X_IC):", X_IC.shape)
print("Boundary condition points at x=0 (X_BC_0):", X_BC_0.shape)
print("Boundary condition points at x=1 (X_BC_1):", X_BC_1.shape)

# Save data to .npy files
np.save('data/X_f.npy', X_f)
np.save('data/X_IC.npy', X_IC)
np.save('data/X_BC_0.npy', X_BC_0)
np.save('data/X_BC_1.npy', X_BC_1)
np.save('data/u_IC.npy', np.sin(np.pi * x_IC))
