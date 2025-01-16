import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import create_model
import json
from scipy.stats import norm

def analytical_solution(x, t, nu=0.01):
    """Analytical solution for Burgers equation (if available)"""
    # This is a simplified version - replace with actual analytical solution if available
    return np.sin(np.pi * x) * np.exp(-nu * np.pi**2 * t)

def compute_pde_residual(model, x, t, nu=0.01):
    """Compute the PDE residual (how well the PDE is satisfied)"""
    X_test = tf.convert_to_tensor(np.stack([t.flatten(), x.flatten()], axis=1), dtype=tf.float32)
    u_pred, u_t, u_x, u_xx = model(X_test)
    residual = u_t + u_pred * u_x - nu * u_xx
    return tf.reshape(residual, x.shape)

def evaluate_model():
    print("\nStarting comprehensive evaluation...")
    
    # Load model and configuration
    with open('model_config.json', 'r') as f:
        model_config = json.load(f)
    print(f"\nModel Configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    model = create_model(**model_config)
    dummy_input = tf.zeros((1, 2))
    _ = model(dummy_input)
    model.load_weights('pinn_model.weights.h5')
    
    # Generate test data
    nx, nt = 100, 100
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, 1, nt)
    X, T = np.meshgrid(x, t)
    X_test = np.stack([T.flatten(), X.flatten()], axis=1)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    
    # Get predictions
    print("\nGenerating predictions...")
    u_pred = model(X_test)[0]
    u_pred = tf.reshape(u_pred, (nt, nx))
    
    # Compute analytical solution
    u_analytical = analytical_solution(X, T)
    
    # Compute error metrics
    error = u_analytical - u_pred.numpy()
    mse = np.mean(error**2)
    mae = np.mean(np.abs(error))
    max_error = np.max(np.abs(error))
    
    print("\nError Metrics:")
    print(f"  Mean Squared Error: {mse:.6f}")
    print(f"  Mean Absolute Error: {mae:.6f}")
    print(f"  Maximum Absolute Error: {max_error:.6f}")
    
    # Compute PDE residuals
    print("\nComputing PDE residuals...")
    residuals = compute_pde_residual(model, X, T)
    mean_residual = tf.reduce_mean(tf.abs(residuals))
    max_residual = tf.reduce_max(tf.abs(residuals))
    print(f"  Mean Absolute Residual: {mean_residual:.6f}")
    print(f"  Maximum Absolute Residual: {max_residual:.6f}")
    
    # Create visualization directory
    os.makedirs('results', exist_ok=True)
    
    # Plot 1: Solution Comparison
    plt.figure(figsize=(20, 5))
    
    plt.subplot(131)
    plt.contourf(T, X, u_pred, levels=50, cmap='rainbow')
    plt.colorbar(label='u(x,t)')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('PINN Solution')
    
    plt.subplot(132)
    plt.contourf(T, X, u_analytical, levels=50, cmap='rainbow')
    plt.colorbar(label='u(x,t)')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Analytical Solution')
    
    plt.subplot(133)
    plt.contourf(T, X, np.abs(error), levels=50, cmap='hot')
    plt.colorbar(label='|Error|')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Absolute Error')
    
    plt.tight_layout()
    plt.savefig('results/solution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Cross-sections at different times
    plt.figure(figsize=(15, 5))
    time_points = [0.0, 0.5, 1.0]
    for t_idx, t_point in enumerate(time_points):
        t_index = int(t_point * (nt-1))
        plt.subplot(1, 3, t_idx+1)
        plt.plot(x, u_pred[t_index], 'b-', label='PINN')
        plt.plot(x, u_analytical[t_index], 'r--', label='Analytical')
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.title(f't = {t_point}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/cross_sections.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: PDE Residuals
    plt.figure(figsize=(10, 8))
    plt.contourf(T, X, residuals, levels=50, cmap='hot')
    plt.colorbar(label='PDE Residual')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('PDE Residuals')
    plt.savefig('results/pde_residuals.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save numerical results
    results = {
        'model_config': model_config,
        'error_metrics': {
            'mse': float(mse),
            'mae': float(mae),
            'max_error': float(max_error),
            'mean_residual': float(mean_residual),
            'max_residual': float(max_residual)
        }
    }
    
    with open('results/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nEvaluation completed. Results saved in 'results' directory:")
    print("  - solution_comparison.png")
    print("  - cross_sections.png")
    print("  - pde_residuals.png")
    print("  - evaluation_results.json")

if __name__ == "__main__":
    evaluate_model()
