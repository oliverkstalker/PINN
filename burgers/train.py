import numpy as np
import tensorflow as tf
from model import create_model, load_data
import tensorflow_probability as tfp
import optuna
from optuna.trial import Trial

def calculate_loss(model, X_f, X_IC, u_IC, X_BC_0, X_BC_1, nu, loss_weights=None):
    # Get predictions and compute individual losses
    u_pred, u_t, u_x, u_xx = model(X_f, training=True)
    
    # Ensure all tensors are of the same type
    u_pred = tf.cast(u_pred, tf.float32)
    u_t = tf.cast(u_t, tf.float32)
    u_x = tf.cast(u_x, tf.float32)
    u_xx = tf.cast(u_xx, tf.float32)
    
    # PDE residual
    residuals = u_t + u_pred * u_x - nu * u_xx
    loss_PDE = tf.reduce_mean(tf.square(residuals))

    # Initial condition loss
    u_IC_pred = model(X_IC, training=True)[0]
    loss_IC = tf.reduce_mean(tf.square(u_IC - u_IC_pred))

    # Boundary conditions loss
    u_BC_0_pred = model(X_BC_0, training=True)[0]
    u_BC_1_pred = model(X_BC_1, training=True)[0]
    loss_BC = tf.reduce_mean(tf.square(u_BC_0_pred)) + tf.reduce_mean(tf.square(u_BC_1_pred))

    # Apply adaptive weights if provided
    if loss_weights is None:
        loss_weights = {'PDE': 1.0, 'IC': 1.0, 'BC': 1.0}
    
    total_loss = (loss_weights['PDE'] * loss_PDE + 
                 loss_weights['IC'] * loss_IC + 
                 loss_weights['BC'] * loss_BC)
    
    return total_loss, {'PDE': float(loss_PDE), 'IC': float(loss_IC), 'BC': float(loss_BC)}

def update_loss_weights(loss_history, window_size=10):
    if len(loss_history) < 2:
        return {'PDE': 1.0, 'IC': 1.0, 'BC': 1.0}
    
    window = min(len(loss_history), window_size)
    recent_losses = loss_history[-window:]
    avg_losses = {k: np.mean([loss[k] for loss in recent_losses]) for k in ['PDE', 'IC', 'BC']}
    
    max_loss = max(avg_losses.values())
    weights = {k: max_loss / (v + 1e-10) for k, v in avg_losses.items()}
    
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    return weights

def train_model(model, X_f, X_IC, u_IC, X_BC_0, X_BC_1, nu, epochs=5000, learning_rate=0.001):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_history = []
    loss_weights = {'PDE': 1.0, 'IC': 1.0, 'BC': 1.0}
    
    for epoch in range(epochs):
        if epoch % 5 == 0 and epoch > 0:
            loss_weights = update_loss_weights(loss_history)
        
        with tf.GradientTape() as tape:
            total_loss, individual_losses = calculate_loss(
                model, X_f, X_IC, u_IC, X_BC_0, X_BC_1, nu, loss_weights)
        
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        loss_history.append(individual_losses)
        
        if epoch % 1 == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch}, Total Loss: {total_loss.numpy():.4f}, '
                  f'Individual Losses: {individual_losses}, '
                  f'Weights: {loss_weights}')
    
    return model, loss_history

def objective(trial: Trial):
    """Optuna objective function for hyperparameter optimization"""
    try:
        # Define hyperparameter search space
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        n_layers = trial.suggest_int('n_layers', 3, 8)
        n_neurons = trial.suggest_int('n_neurons', 16, 128)
        activation = trial.suggest_categorical('activation', ['tanh', 'sigmoid'])
        
        # Create and initialize model
        model = create_model(n_layers, n_neurons, activation)
        
        # Load and preprocess data
        X_f, X_IC, X_BC_0, X_BC_1, u_IC = load_data()
        X_f = tf.convert_to_tensor(X_f, dtype=tf.float32)
        X_IC = tf.convert_to_tensor(X_IC, dtype=tf.float32)
        u_IC = tf.convert_to_tensor(u_IC, dtype=tf.float32)
        X_BC_0 = tf.convert_to_tensor(X_BC_0, dtype=tf.float32)
        X_BC_1 = tf.convert_to_tensor(X_BC_1, dtype=tf.float32)
        
        # Train with suggested hyperparameters
        nu = 0.01
        epochs = 50  # Reduced from 100 to 30
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                total_loss, individual_losses = calculate_loss(
                    model, X_f, X_IC, u_IC, X_BC_0, X_BC_1, nu
                )
            
            gradients = tape.gradient(total_loss, model.trainable_variables)
            tf.keras.optimizers.Adam(learning_rate).apply_gradients(
                zip(gradients, model.trainable_variables)
            )
            
            # Report intermediate value for pruning
            trial.report(float(total_loss), epoch)
            
            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Early stopping
            if total_loss < best_loss:
                best_loss = total_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 5 == 0:
                print(f"Trial {trial.number}, Epoch {epoch}, Loss: {float(total_loss):.4f}")
        
        return float(best_loss)
    
    except Exception as e:
        print(f"Trial failed with error: {str(e)}")
        return float('inf')

if __name__ == "__main__":
    # Create Optuna study with pruning
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1
        )
    )
    
    # Run hyperparameter optimization
    n_trials = 50
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save the best hyperparameters and model
    best_params = study.best_params
    model_config = {
        'n_layers': best_params['n_layers'],
        'n_neurons': best_params['n_neurons'],
        'activation': best_params['activation']
    }
    
    # Save model configuration
    import json
    with open('model_config.json', 'w') as f:
        json.dump(model_config, f)
    
    # Load data for final training
    X_f, X_IC, X_BC_0, X_BC_1, u_IC = load_data()
    X_f = tf.convert_to_tensor(X_f, dtype=tf.float32)
    X_IC = tf.convert_to_tensor(X_IC, dtype=tf.float32)
    u_IC = tf.convert_to_tensor(u_IC, dtype=tf.float32)
    X_BC_0 = tf.convert_to_tensor(X_BC_0, dtype=tf.float32)
    X_BC_1 = tf.convert_to_tensor(X_BC_1, dtype=tf.float32)
    
    # Create and train final model with best parameters
    final_model = create_model(**model_config)
    
    # Train final model with best hyperparameters
    nu = 0.01
    epochs = 500
    print("\nTraining final model with best parameters...")
    trained_model, loss_history = train_model(
        final_model, X_f, X_IC, u_IC, X_BC_0, X_BC_1, 
        nu, epochs, best_params['learning_rate']
    )
    
    # Save the trained model weights
    trained_model.save_weights('pinn_model.weights.h5')
    
    print("\nTraining completed. Model saved.")
