import numpy as np
import tensorflow as tf
from model import create_model, load_data

def train_model(model, X_f, X_IC, u_IC, X_BC_0, X_BC_1, epochs, learning_rate):
    # Define the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = model.calculate_loss(X_f, X_IC, u_IC, X_BC_0, X_BC_1)
        
        # Compute gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # Apply gradients
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Print loss every 100 epochs
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.numpy()}')

if __name__ == "__main__":
    # Load the data
    X_f, X_IC, X_BC_0, X_BC_1, u_IC = load_data()

    # Convert data to TensorFlow tensors
    X_f = tf.convert_to_tensor(X_f, dtype=tf.float32)
    X_IC = tf.convert_to_tensor(X_IC, dtype=tf.float32)
    u_IC = tf.convert_to_tensor(u_IC, dtype=tf.float32)
    X_BC_0 = tf.convert_to_tensor(X_BC_0, dtype=tf.float32)
    X_BC_1 = tf.convert_to_tensor(X_BC_1, dtype=tf.float32)

    # Create the model
    model = create_model()

    # Print the model summary to verify the architecture
    model.build(input_shape=(None, 2))
    model.summary()

    # Train the model
    epochs = 100
    learning_rate = 0.001
    train_model(model, X_f, X_IC, u_IC, X_BC_0, X_BC_1, epochs, learning_rate)
