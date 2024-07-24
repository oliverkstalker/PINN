import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class Network:
    @classmethod
    def build(cls, num_inputs=2 ,layers=[16, 32, 64], activation='tanh', num_outputs=1):
        inputs = tf.keras.layers.Input(shape=(num_inputs,))
        x = inputs
        for layer in layers:
            x = tf.keras.layers.Dense(layer, activation=activation)(x)
        outputs = tf.keras.layers.Dense(num_outputs)(x)
        return tf.keras.models.Model(inputs=inputs, outputs=outputs)

class GradientLayer(tf.keras.layers.Layer):
    def __init__(self, model, **kwargs):
        self.model = model
        super().__init__(**kwargs)
    def call(self, x):
        try:
            with tf.GradientTape(persistent=True) as g:
                g.watch(x)
                with tf.GradientTape(persistent=True) as gg:
                    gg.watch(x)
                    u = self.model(x)
                du_dtx = gg.batch_jacobian(u, x)
                du_dt = du_dtx[..., 0]
                du_dx = du_dtx[..., 1]
            d2u_dx2 = g.batch_jacobian(du_dx, x)[..., 1]
            return u, du_dt, du_dx, d2u_dx2
        except Exception as e:
            print(f"Error in GradientLayer call: {e}")
            print(f"x shape: {x.shape}")
            print(f"u shape: {u.shape if 'u' in locals() else 'undefined'}")
            raise e

    def summary(self):
        self.model.summary()
    def save_weights(self, path):
        self.model.save_weights(path)
    def load_weights(self, path):
        self.model.load_weights(path)

def create_model():
    network = Network.build()
    gradient_layer = GradientLayer(network)
    return gradient_layer

def load_data():
    X_f = np.load('data/X_f.npy')
    X_IC = np.load('data/X_IC.npy')
    X_BC_0 = np.load('data/X_BC_0.npy')
    X_BC_1 = np.load('data/X_BC_1.npy')
    u_IC = np.load('data/u_IC.npy')
    return X_f, X_IC, X_BC_0, X_BC_1, u_IC
