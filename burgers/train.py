import numpy as np
import tensorflow as tf
from model import create_model, load_data
from scipy.optimize import fmin_l_bfgs_b

def calculate_loss(model, X_f, X_IC, u_IC, X_BC_0, X_BC_1, nu):
    u_pred, u_t, u_x, u_xx = model(X_f)
    residuals = u_t + u_pred * u_x - nu * u_xx
    loss_PDE = tf.reduce_mean(tf.square(residuals))

    u_IC_pred, _, _, _ = model(X_IC)
    loss_IC = tf.reduce_mean(tf.square(u_IC - u_IC_pred))

    u_BC_0_pred, _, _, _ = model(X_BC_0)
    u_BC_1_pred, _, _, _ = model(X_BC_1)
    loss_BC_0 = tf.reduce_mean(tf.square(u_BC_0_pred))
    loss_BC_1 = tf.reduce_mean(tf.square(u_BC_1_pred))
    loss_BC = loss_BC_0 + loss_BC_1

    total_loss = loss_PDE + loss_IC + loss_BC
    return total_loss

class L_BFGS_B:
    def __init__(self, model, x_train, y_train, factr=1e7, m=50, maxls=50, maxiter=5000):
        self.model = model
        self.x_train = [tf.constant(x, dtype=tf.float32) for x in x_train]
        self.y_train = [tf.constant(y, dtype=tf.float32) for y in y_train]
        self.factr = factr
        self.m = m
        self.maxls = maxls
        self.maxiter = maxiter
        self.metrics = ['loss']
        self.progbar = tf.keras.callbacks.ProgbarLogger()

    def set_weights(self, flat_weights):
        shapes = [w.shape for w in self.model.weights]
        split_weights = np.split(flat_weights, np.cumsum([np.prod(s) for s in shapes])[:-1])
        reshaped_weights = [w.reshape(s) for w, s in zip(split_weights, shapes)]
        self.model.set_weights(reshaped_weights)

    def tf_evaluate(self, x, y):
        # Concatenate inputs
        x = tf.concat(x, axis=0)
        y = tf.concat(y, axis=0)
        with tf.GradientTape() as g:
            u_pred, _, _, _ = self.model(x)
            y_true = tf.reshape(y, u_pred.shape)  # Ensure y_true has the correct shape
            loss = tf.reduce_mean(tf.keras.losses.mse(y_true, u_pred))
        grads = g.gradient(loss, self.model.trainable_variables)
        return loss, grads

    def evaluate(self, weights):
        self.set_weights(weights)
        loss, grads = self.tf_evaluate(self.x_train, self.y_train)
        loss = loss.numpy().astype('float64')
        grads = np.concatenate([g.numpy().flatten() for g in grads]).astype('float64')
        return loss, grads

    def callback(self, weights):
        self.progbar.on_batch_begin(0)
        loss, _ = self.evaluate(weights)
        self.progbar.on_batch_end(0, logs=dict(zip(self.metrics, [loss])))

    def fit(self):
        initial_weights = np.concatenate([w.flatten() for w in self.model.get_weights()])
        print('Optimizer: L-BFGS-B (maxiter={})'.format(self.maxiter))
        self.progbar.on_train_begin()
        self.progbar.on_epoch_begin(1)
        fmin_l_bfgs_b(func=self.evaluate, x0=initial_weights,
                      factr=self.factr, m=self.m, maxls=self.maxls, maxiter=self.maxiter,
                      callback=self.callback)
        self.progbar.on_epoch_end(1)
        self.progbar.on_train_end()

if __name__ == "__main__":
    X_f, X_IC, X_BC_0, X_BC_1, u_IC = load_data()

    X_f = tf.convert_to_tensor(X_f, dtype=tf.float32)
    X_IC = tf.convert_to_tensor(X_IC, dtype=tf.float32)
    u_IC = tf.convert_to_tensor(u_IC, dtype=tf.float32)
    X_BC_0 = tf.convert_to_tensor(X_BC_0, dtype=tf.float32)
    X_BC_1 = tf.convert_to_tensor(X_BC_1, dtype=tf.float32)

    model = create_model()

    model.build(input_shape=(None, 2))
    model.summary()

    # Train with Adam optimizer
    epochs = 5
    initial_learning_rate = 0.001
    nu = 0.01
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = calculate_loss(model, X_f, X_IC, u_IC, X_BC_0, X_BC_1, nu)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        print(f'Epoch {epoch}, Loss: {loss.numpy()}')

    model.save_weights('pinn_model.weights.h5')

    # Fine-tuning with L-BFGS-B optimizer
    x_train = [X_f, X_IC, X_BC_0, X_BC_1]
    y_train = [u_IC, u_IC, u_IC]  # Adjust this according to your data structure
    lbfgs = L_BFGS_B(model, x_train, y_train, factr=1e7, m=50, maxls=50, maxiter=5000)
    lbfgs.fit()

    # Save the trained model weights after fine-tuning
    model.save_weights('pinn_model.weights.h5')
