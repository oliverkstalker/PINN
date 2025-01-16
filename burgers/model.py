import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class PINN(tf.keras.Model):
    def __init__(self, n_layers=4, n_neurons=64, activation='tanh'):
        super().__init__()
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.activation = activation
        
        # Define network layers
        self.hidden_layers = []
        for _ in range(n_layers):
            self.hidden_layers.append(tf.keras.layers.Dense(
                n_neurons, 
                activation=activation,
                kernel_initializer='glorot_normal'
            ))
        self.output_layer = tf.keras.layers.Dense(1)

    def build(self, input_shape):
        # Build the model by passing a dummy input
        dummy_input = tf.keras.layers.Input(shape=input_shape[1:])
        x = dummy_input
        for layer in self.hidden_layers:
            x = layer(x)
        _ = self.output_layer(x)
        super().build(input_shape)

    def call(self, x, training=None):
        t, x_coord = tf.split(x, 2, axis=1)
        
        # Ensure inputs are treated as trainable tensors
        t = tf.convert_to_tensor(t, dtype=tf.float32)
        x_coord = tf.convert_to_tensor(x_coord, dtype=tf.float32)
        
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(t)
            tape2.watch(x_coord)
            
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(t)
                tape1.watch(x_coord)
                
                # Forward pass
                inputs = tf.concat([t, x_coord], axis=1)
                x = inputs
                for layer in self.hidden_layers:
                    x = layer(x)
                u = self.output_layer(x)
            
            # First derivatives
            u_t = tape1.gradient(u, t)
            u_x = tape1.gradient(u, x_coord)
            
        # Second derivative
        u_xx = tape2.gradient(u_x, x_coord)
        
        # Clean up
        del tape1, tape2
        
        # Ensure no None values
        u_t = tf.zeros_like(u) if u_t is None else u_t
        u_x = tf.zeros_like(u) if u_x is None else u_x
        u_xx = tf.zeros_like(u) if u_xx is None else u_xx
        
        return u, u_t, u_x, u_xx

def create_model(n_layers=4, n_neurons=64, activation='tanh'):
    model = PINN(n_layers, n_neurons, activation)
    # Build the model with a dummy input
    dummy_input = tf.zeros((1, 2))
    _ = model(dummy_input)
    return model

def load_data():
    X_f = np.load('data/X_f.npy')
    X_IC = np.load('data/X_IC.npy')
    X_BC_0 = np.load('data/X_BC_0.npy')
    X_BC_1 = np.load('data/X_BC_1.npy')
    u_IC = np.load('data/u_IC.npy')
    return X_f, X_IC, X_BC_0, X_BC_1, u_IC