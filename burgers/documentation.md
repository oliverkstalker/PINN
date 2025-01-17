# Comprehensive Documentation for the PINN Codebase

## Overview

This codebase implements a Physics-Informed Neural Network (PINN) to solve the Burgers' equation, a fundamental partial differential equation used in fluid mechanics and nonlinear dynamics. The project combines deep learning with physical constraints to predict solutions that adhere to the equation's governing principles.

The codebase consists of modular scripts for data preparation, model definition, training, evaluation, and overall workflow orchestration. Each component is designed to ensure high accuracy, efficient computation, and adherence to the physics of the Burgers' equation.

---

## Project Structure

```plaintext
burgers/
├── data_utils.py       # Data generation for collocation, boundary, and initial conditions
├── model.py            # Definition of the PINN model and its gradient calculations
├── train.py            # Training process with loss computation and hyperparameter optimization
├── evaluate.py         # Model evaluation and comparison against analytical solutions
├── run.py              # Main script to execute the entire workflow
├── results/            # Directory to store outputs (plots, metrics, and model weights)
│   ├── model_config.json
│   ├── pinn_model.weights.h5
│   ├── solution_comparison.png
│   ├── cross_sections.png
│   ├── pde_residuals.png
│   └── evaluation_results.json
└── documentation.md    # Project documentation
```

## File Descriptions
### `data_utils.py`

#### Purpose
This script generates the datasets required for training the PINN, including collocation points, initial condition points, and boundary condition points.

#### Key Features
1. **Stratified Sampling**:
   - Ensures uniform distribution across the domain for collocation points.
2. **Dynamic Initial Conditions**:
   - Adds variability to the initial conditions with small perturbations.
3. **Perturbed Boundary Conditions**:
   - Introduces noise to boundary condition points for robustness.

#### Outputs
- `X_f.npy`: Collocation points.
- `X_IC.npy`: Initial condition points.
- `X_BC_0.npy` and `X_BC_1.npy`: Boundary condition points.
- `u_IC.npy`: Values at the initial condition.
___
### `model.py`

#### Purpose
Defines the PINN architecture and includes functionality for gradient computation, essential for solving the Burgers' equation while adhering to its governing physical laws.

#### Key Features
1. **Model Architecture**:
   - Fully connected feed-forward network with configurable layers and neurons.
   - Supports hyperparameter tuning for activation functions and architecture design.

2. **Gradient Computation**:
   - Uses TensorFlow's `GradientTape` to compute first- and second-order derivatives of the predicted solution.

3. **Utility Functions**:
   - `create_model`: Initializes the PINN with a specified architecture.
   - `load_data`: Loads the preprocessed datasets from `.npy` files.

#### Outputs
- Trained model with gradient computation capabilities for enforcing physics-based constraints.
___
### `train.py`

#### Purpose
Manages the training pipeline for the PINN, including loss calculation, optimization, and hyperparameter tuning with Bayesian optimization.

#### Key Features
1. **Loss Calculation**:
   - Computes separate losses for PDE residuals, initial conditions, and boundary conditions.
   - Supports adaptive loss weighting to balance contributions dynamically.

2. **Training Loop**:
   - Optimizes the model using the Adam optimizer and dynamically updates loss weights.

3. **Hyperparameter Optimization**:
   - Integrates Optuna for tuning key parameters such as learning rate, number of layers, neurons, and activation functions.

#### Outputs
- Trained model weights saved to `results/pinn_model.weights.h5`.
- Hyperparameter optimization results stored in `model_config.json`.
___
### `evaluate.py`

#### Purpose
Evaluates the trained PINN by computing error metrics, analyzing PDE residuals, and visualizing the solution and errors.

#### Key Features
1. **Error Metrics**:
   - Computes Mean Squared Error (MSE), Mean Absolute Error (MAE), and Maximum Error.

2. **Residual Analysis**:
   - Measures how well the model satisfies the PDE constraints.

3. **Visualization**:
   - Produces 3D plots of the predicted solution, analytical solution, and residuals.
   - Saves cross-sectional comparisons at key time points.

#### Outputs
- `solution_comparison.png`: Visual comparison of the predicted and analytical solutions.
- `cross_sections.png`: Cross-sectional views of the solutions at different time points.
- `pde_residuals.png`: Heatmap of the PDE residuals.
- `evaluation_results.json`: Numerical metrics summarizing model performance.

___
### `run.py`

#### Purpose
Automates the workflow by sequentially running the data generation, training, and evaluation scripts.

#### Key Features
1. **Workflow Automation**:
   - Calls each script in sequence using `subprocess.run`.
   - Ensures data dependencies are correctly passed between steps.

2. **Error Handling**:
   - Stops execution if any step fails to ensure data consistency.

#### Outputs
- Results from all stages (data generation, training, evaluation) saved to the `results` directory.
___
## Results Directory Documentation

The `results/` directory contains the outputs generated during the training and evaluation of the Physics-Informed Neural Network (PINN) for solving the Burgers' equation. These results provide insights into the model's performance, including error metrics, hyperparameter optimization details, and visualizations of the predicted solution and residuals.

### 1. `solution_comparison.png`

#### Description
- A visualization comparing the PINN-predicted solution, the analytical solution, and their absolute error.
- **Subplots**:
  1. **PINN Solution**: The solution predicted by the PINN.
  2. **Analytical Solution**: The true solution (if available) computed analytically.
  3. **Absolute Error**: The magnitude of the error between the predicted and analytical solutions.

#### Purpose
- Provides a visual comparison of the PINN's accuracy relative to the analytical solution.
- Highlights regions with significant prediction errors.


### 2. `cross_sections.png`

#### Description
- Line plots showing cross-sectional comparisons of the PINN-predicted solution and the analytical solution at different time points (`t=0.0`, `t=0.5`, `t=1.0`).
- Each subplot corresponds to a specific time point.

#### Purpose
- Offers a detailed view of the PINN's performance along the spatial dimension (`x`) for various times.
- Helps identify temporal variations in prediction accuracy.


### 3. `pde_residuals.png`

#### Description
- A heatmap illustrating the residuals of the Burgers' equation across the domain.
- Residuals are computed as the difference between the left-hand side and the right-hand side of the PDE.

#### Purpose
- Visualizes how well the PINN satisfies the Burgers' equation.
- Highlights regions where the PINN struggles to adhere to the physics.



