import numpy as np
import tensorflow as tf
from model import create_model, load_data

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

def train_model(model, X_f, X_IC, u_IC, X_BC_0, X_BC_1, nu, epochs=5000, learning_rate=0.001):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = calculate_loss(model, X_f, X_IC, u_IC, X_BC_0, X_BC_1, nu)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch}, Loss: {loss.numpy()}')
    return model

if __name__ == "__main__":
    # Load and preprocess data
    X_f, X_IC, X_BC_0, X_BC_1, u_IC = load_data()

    # Convert data to tensors
    X_f = tf.convert_to_tensor(X_f, dtype=tf.float32)
    X_IC = tf.convert_to_tensor(X_IC, dtype=tf.float32)
    u_IC = tf.convert_to_tensor(u_IC, dtype=tf.float32)
    X_BC_0 = tf.convert_to_tensor(X_BC_0, dtype=tf.float32)
    X_BC_1 = tf.convert_to_tensor(X_BC_1, dtype=tf.float32)

    # Create and build the model
    model = create_model()
    model.build(input_shape=(None, 2))
    model.summary()

    # Set hyperparameters
    nu = 0.01
    epochs = 15
    learning_rate = 0.001

    # Train the model
    trained_model = train_model(model, X_f, X_IC, u_IC, X_BC_0, X_BC_1, nu, epochs, learning_rate)

    # Save the trained model weights
    trained_model.save_weights('results/pinn_model.weights.h5')
