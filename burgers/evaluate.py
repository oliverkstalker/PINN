import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model import create_model, load_data

def generate_test_data(num_points=100):
    x = np.linspace(0, 1, num_points)
    t = np.linspace(0, 1, num_points)
    X, T = np.meshgrid(x, t)
    X_test = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    print(f"Generated test data X_test shape: {X_test.shape}, X shape: {X.shape}, T shape: {T.shape}")
    return X_test, X, T

def evaluate_model(model, X_test):
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    u_pred, _, _, _ = model(X_test)
    print(f"Evaluated model predictions u_pred shape: {u_pred.shape}")
    return u_pred.numpy()

def plot_results(X, T, u_pred, residuals):
    fig = plt.figure(figsize=(12, 6))
    
    # Predicted Solution
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, T, u_pred.reshape(X.shape), cmap='viridis')
    ax1.set_title('Predicted Solution')
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_zlabel('u')
    
    # Residuals
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, T, residuals.reshape(X.shape), cmap='viridis')
    ax2.set_title('Residuals')
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_zlabel('Residual')

    plt.savefig('results/results.png')

def compute_residuals(model, X_test, nu):
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    
    u_pred, u_t, u_x, u_xx = model(X_test)
    residuals = u_t + u_pred * u_x - nu * u_xx
    print(f"Computed residuals shape: {residuals.shape}")
    return residuals.numpy()

if __name__ == "__main__":
    X_f, X_IC, X_BC_0, X_BC_1, u_IC = load_data()

    model = create_model()

    model.load_weights('pinn_model.weights.h5')

    X_test, X, T = generate_test_data()

    u_pred = evaluate_model(model, X_test)

    nu = 0.01
    residuals = compute_residuals(model, X_test, nu)
    print(f'Mean residual error: {np.mean(np.abs(residuals))}')

    plot_results(X, T, u_pred, residuals)
