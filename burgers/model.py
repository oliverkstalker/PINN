import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class BurgersPINN(models.Model):
    def __init__(self, layers_config, nu):
        super(BurgersPINN, self).__init__()
        self.hidden_layers = []
        self.nu = nu
        # Input Layer
        self.input_layer = layers.InputLayer(input_shape=(2,))
        
        # Hidden Layers
        for units in layers_config:
            self.hidden_layers.append(layers.Dense(units, activation='tanh'))
        
        # Output Layer
        self.output_layer = layers.Dense(1)

    def build(self, input_shape):
        # Build the layers
        super(BurgersPINN, self).build(input_shape)
        for layer in self.hidden_layers:
            layer.build(input_shape)
            input_shape = (input_shape[0], layer.units)
        self.output_layer.build(input_shape)
    
    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

    def calculate_loss(self, X_f, X_IC, u_IC, X_BC_0, X_BC_1):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X_f)
            u = self.call(X_f)
            u_x = tape.gradient(u, X_f)[:, 0]
            u_t = tape.gradient(u, X_f)[:, 1]
        u_xx = tape.gradient(u_x, X_f)[:, 0]
        del tape

        # Compute PDE residual
        R = u_t + u * u_x - self.nu * u_xx
        loss_PDE = tf.reduce_mean(tf.square(R))

        # Compute initial condition loss
        u_IC_pred = self.call(X_IC)
        loss_IC = tf.reduce_mean(tf.square(u_IC - u_IC_pred))

        # Compute boundary condition loss
        u_BC_0 = self.call(X_BC_0)
        u_BC_1 = self.call(X_BC_1)
        loss_BC_0 = tf.reduce_mean(tf.square(u_BC_0))
        loss_BC_1 = tf.reduce_mean(tf.square(u_BC_1))
        loss_BC = loss_BC_0 + loss_BC_1

        # Total loss
        total_loss = loss_PDE + loss_IC + loss_BC
        return total_loss

def create_model():
    # Define the configuration for the hidden layers
    layers_config = [50, 50]  # Example configuration: two hidden layers with 50 neurons each
    nu = 0.01  # Viscosity parameter for Burgers' equation
    model = BurgersPINN(layers_config, nu)
    return model

def load_data():
    X_f = np.load('data/X_f.npy')
    X_IC = np.load('data/X_IC.npy')
    X_BC_0 = np.load('data/X_BC_0.npy')
    X_BC_1 = np.load('data/X_BC_1.npy')
    u_IC = np.load('data/u_IC.npy')
    return X_f, X_IC, X_BC_0, X_BC_1, u_IC
