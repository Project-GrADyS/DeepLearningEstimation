"""
Model Training Module for Distributed Formation Control

This module is responsible for training a deep neural network model to predict
communication parameters based on formation characteristics.

Key Features:
- Data loading and preprocessing
- Model architecture definition
- Training pipeline with early stopping
- Model evaluation and metrics calculation
- Model saving and loading
- Flexible feature/target selection

Dependencies:
    - numpy: Numerical computations
    - pandas: Data manipulation
    - torch: Deep learning framework
    - sklearn: Metrics and utilities
    - matplotlib: Visualization

Author: Laércio Lucchesi
Date: 2025-03-30
Version: 1.0
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import time
from typing import Tuple, List, Dict, Optional
from feature_importance import compute_feature_importance
# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class FormationDataset(Dataset):
    """
    Custom Dataset class for formation control data.
    """
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """
        Initialize the dataset.
        
        Args:
            features (np.ndarray): Input features
            targets (np.ndarray): Target values
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        return self.features[idx], self.targets[idx]

class FormationModel(nn.Module):
    """
    Deep neural network model for formation control prediction.
    """
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, dropout_rate: float = 0.0):
        """
        Initialize the model.
        
        Args:
            input_size (int): Number of input features
            hidden_sizes (List[int]): List of hidden layer sizes
            output_size (int): Number of output targets
            dropout_rate (float): Dropout rate for regularization
        """
        super(FormationModel, self).__init__()
        
        # Build the network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LeakyReLU(0.1))  # Using LeakyReLU instead of ReLU
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        # Combine all layers
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return self.network(x)

def get_variable_roles() -> Dict[str, str]:
    """
    Ask the user to define the role of each variable.
    
    Returns:
        Dict[str, str]: Dictionary mapping variable names to their roles ('feature', 'target', or 'none')
    """
    variables = ['size_formation', 'comm_delay', 'comm_range', 'comm_failure']
    
    # Define the time periods for each metric type
    time_periods = ['030s', '1min', '2min', '3min', '4min', '5min', 'multi_period']
    
    # Define the metric types
    metric_types = ['shape_error', 'leader_off', 'presence_error']
    
    # Ask the user to select the time period (only once)
    print("\nChoose the time period to use for all metrics:")
    for i, period in enumerate(time_periods):
        print(f"{i}. {period}")
    
    while True:
        choice = input(f"Enter option number (0-{len(time_periods)-1}): ")
        if choice.isdigit() and 0 <= int(choice) < len(time_periods):
            break
        else:
            print(f"Invalid input. Please enter a number between 0 and {len(time_periods)-1}.")
    
    # Set selected time period
    selected_period = time_periods[int(choice)]
    print(f"\nSelected time period: {selected_period}")
    
    # Create the selected metrics for each metric type
    selected_metrics = {}
    
    if selected_period == 'multi_period':
        # If multi_period is selected, include all periods for each metric type
        for metric_type in metric_types:
            for period in time_periods[:-1]:  # Exclude 'multi_period' itself
                metric_name = f'{metric_type}_avg_{period}'
                selected_metrics[metric_name] = metric_name
                print(f"Including {metric_name} as a separate metric")
    else:
        # Otherwise, use only the selected period
        for metric_type in metric_types:
            selected_metric = f'{metric_type}_avg_{selected_period}'
            selected_metrics[metric_type] = selected_metric
            print(f"Using {selected_metric} for {metric_type}")
    
    # Initialize roles dictionary with selected metrics
    roles = {metric: 'none' for metric in selected_metrics.values()}
    
    # Ask for role of each variable
    print("\nFor each variable, choose its role:")
    print("0. Not used")
    print("1. Feature (input to the model)")
    print("2. Target (output to predict)")
    
    # Track if at least one target is selected
    has_target = False
    
    for var in variables:
        print(f"\n{var}:")
        while True:
            role_choice = input("Enter option number (0-2): ")
            if role_choice in ["0", "1", "2"]:
                break
            else:
                print("Invalid input. Please enter 0, 1, or 2.")
        
        role = {
            "0": "none",
            "1": "feature",
            "2": "target"
        }[role_choice]
        
        roles[var] = role
        print(f"{var} will be used as: {role}")
        
        # Check if this variable is a target
        if role == "target":
            has_target = True
    
    # Ask for role of each selected metric
    for metric_name in selected_metrics.values():
        print(f"\n{metric_name}:")
        while True:
            role_choice = input("Enter option number (0-2): ")
            if role_choice in ["0", "1", "2"]:
                break
            else:
                print("Invalid input. Please enter 0, 1, or 2.")
        
        role = {
            "0": "none",
            "1": "feature",
            "2": "target"
        }[role_choice]
        
        roles[metric_name] = role
        print(f"{metric_name} will be used as: {role}")
        
        # Check if this variable is a target
        if role == "target":
            has_target = True
    
    # If no targets are selected, ask the user to select at least one
    if not has_target:
        print("\nWARNING: No targets selected. You must select at least one target.")
        print("Please select a target variable:")
        
        # List all variables again
        all_vars = variables + list(selected_metrics.values())
        for i, var in enumerate(all_vars):
            print(f"{i+1}. {var}")
        
        while True:
            try:
                target_choice = int(input("Enter the number of the variable to use as target: "))
                if 1 <= target_choice <= len(all_vars):
                    target_var = all_vars[target_choice-1]
                    roles[target_var] = "target"
                    print(f"{target_var} will be used as target.")
                    break
                else:
                    print(f"Invalid input. Please enter a number between 1 and {len(all_vars)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    return roles

def load_data(file_path: str, roles: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess the dataset.
    
    Args:
        file_path (str): Path to the dataset CSV file
        roles (Dict[str, str]): Dictionary mapping variable names to their roles
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            Training features, training targets, validation features, validation targets, 
            test features, test targets
    """
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Select features and targets based on roles
    features = [var for var, role in roles.items() if role == 'feature']
    targets = [var for var, role in roles.items() if role == 'target']
    
    # Verify that we have at least one feature and one target
    if not features:
        raise ValueError("No features selected. Please select at least one feature.")
    if not targets:
        raise ValueError("No targets selected. Please select at least one target.")
    
    print(f"\nFeatures: {features}")
    print(f"Targets: {targets}")
    
    # Split data by dataset type
    train_data = data[data['dataset_type'] == 'training']
    val_data = data[data['dataset_type'] == 'validation']
    test_data = data[data['dataset_type'] == 'test']
    
    total_data_size = len(data) # size of the entire dataset in number of samples
    print(f"\nDataset sizes")
    print(f"Training:   {len(train_data)} samples ({len(train_data) / total_data_size * 100:.1f}%)")
    print(f"Validation:  {len(val_data)} samples ({len(val_data) / total_data_size * 100:.1f}%)")
    print(f"Test:        {len(test_data)} samples ({len(test_data) / total_data_size * 100:.1f}%)")
    print(" ")
    
    # Extract features and targets
    X_train = train_data[features].values
    y_train = train_data[targets].values
    
    X_val = val_data[features].values
    y_val = val_data[targets].values
    
    X_test = test_data[features].values
    y_test = test_data[targets].values
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    num_epochs: int,
    patience: int,
    device: torch.device
) -> Tuple[List[float], List[float], int]:
    """
    Train the model with early stopping.
    
    Args:
        model (nn.Module): The neural network model
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        criterion (nn.Module): Loss function
        optimizer (optim.Optimizer): Optimizer
        scheduler (ReduceLROnPlateau): Learning rate scheduler
        num_epochs (int): Maximum number of epochs
        patience (int): Early stopping patience
        device (torch.device): Device to use for training
        
    Returns:
        Tuple[List[float], List[float], int]: 
            Training losses, validation losses, best epoch
    """
    model = model.to(device)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return train_losses, val_losses, best_epoch

def evaluate_model(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: torch.device, targets: List[str]) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate the model on the test set.
    
    Args:
        model (nn.Module): The trained model
        test_loader (DataLoader): DataLoader for test data
        criterion (nn.Module): Loss function
        device (torch.device): Device to use for evaluation
        targets (List[str]): List of target variable names
        
    Returns:
        Tuple[float, Dict[str, float]]: Test loss and metrics dictionary
    """
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets_tensor in test_loader:
            features, targets_tensor = features.to(device), targets_tensor.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets_tensor)
            test_loss += loss.item()
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets_tensor.cpu().numpy())
    
    test_loss /= len(test_loader)
    
    # Convert predictions and targets to numpy arrays
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Calculate metrics for each target
    metrics = {}
    
    # Check if we have a single target (1D array) or multiple targets (2D array)
    if len(all_targets.shape) == 1 or all_targets.shape[1] == 1:
        # Single target case
        mse = mean_squared_error(all_targets.ravel(), all_preds.ravel())
        r2 = r2_score(all_targets.ravel(), all_preds.ravel())
        metrics[f'{targets[0]}_mse'] = mse
        metrics[f'{targets[0]}_r2'] = r2
    else:
        # Multiple targets case
        for i, target_name in enumerate(targets):
            if i < all_targets.shape[1]:  # Ensure index is within bounds
                mse = mean_squared_error(all_targets[:, i], all_preds[:, i])
                r2 = r2_score(all_targets[:, i], all_preds[:, i])
                metrics[f'{target_name}_mse'] = mse
                metrics[f'{target_name}_r2'] = r2
    
    return test_loss, metrics

def plot_losses(train_losses: List[float], val_losses: List[float], save_path: Optional[str] = None):
    """
    Plot training and validation losses.
    
    Args:
        train_losses (List[float]): Training losses
        val_losses (List[float]): Validation losses
        save_path (Optional[str]): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Loss plot saved to {save_path}")
    
    plt.show()

def main():
    """
    Main function to execute the model training pipeline.
    """
    # Configuration
    DATASET_PATH = 'dataset.csv'
    HIDDEN_SIZES = [256, 128, 64, 32]  # Hidden layer sizes
    DROPOUT_RATE = 0.1  # Dropout rate
    BATCH_SIZE = 64 # Batch size
    NUM_EPOCHS = 1000  # Number of epochs
    PATIENCE = 100  # Early stopping patience
    LEARNING_RATE = 0.001  # Learning rate
    WEIGHT_DECAY = 0.0001  # Weight decay
    
    # Get variable roles from user
    roles = get_variable_roles()
    
    # Create output directory
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading data from {DATASET_PATH}...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(DATASET_PATH, roles)
    
    # Create datasets
    train_dataset = FormationDataset(X_train, y_train)
    val_dataset = FormationDataset(X_val, y_val)
    test_dataset = FormationDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Create model
    input_size = X_train.shape[1]  # Number of input features
    output_size = y_train.shape[1]  # Number of output targets
    model = FormationModel(input_size, HIDDEN_SIZES, output_size, DROPOUT_RATE)
    
    # Define loss function and optimizer
    criterion = nn.SmoothL1Loss()  # Using SmoothL1Loss instead of MSELoss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Define learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Train model
    print("Starting model training...")
    start_time = time.time()
    train_losses, val_losses, best_epoch = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS, PATIENCE, device
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Get target names for evaluation
    targets = [var for var, role in roles.items() if role == 'target']
    
    # Create a shorter filename based on the type of model training
    model_type = "preliminary"
    
    # Plot losses
    plot_losses(train_losses, val_losses, f'plots/loss_plot_{model_type}.png')
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    test_loss, metrics = evaluate_model(model, test_loader, criterion, device, targets)
    
    # Print test results
    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.6f}")
    print("\nMetrics for each target variable:")
    for target in targets:
        print(f"{target}:")
        print(f"  MSE: {metrics[f'{target}_mse']:.6f}")
        print(f"  R² Score: {metrics[f'{target}_r2']:.6f}")
    
    # Save model with configuration in filename
    model_path = f'models/model_{model_type}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch,
        'test_loss': test_loss,
        'metrics': metrics,
        'config': {
            'input_size': input_size,
            'hidden_sizes': HIDDEN_SIZES,
            'output_size': output_size,
            'dropout_rate': DROPOUT_RATE,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'roles': roles
        }
    }, model_path)
    print(f"\nModel saved to {model_path}")

    # Feature Importance (Captum or SHAP)
    print("\nCalculating feature importance...")

    # Convert test data to tensor
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Get feature names
    feature_names = [var for var, role in roles.items() if role == 'feature']
    
    # Get target names
    target_names = [var for var, role in roles.items() if role == 'target']
    
    # Ask the user which method to use
    print("\nWhich method would you like to use for feature importance calculation?")
    print("1. Captum (DeepLiftShap)")
    print("2. SHAP (KernelExplainer)")
    
    while True:
        try:
            method_choice = int(input("Enter option number (1-2): "))
            if method_choice in [1, 2]:
                break
            else:
                print("Invalid input. Please enter 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Convert choice to method name
    method = "captum" if method_choice == 1 else "shap"
    print(f"Using {method.upper()} for feature importance calculation.")
    
    # Calculate feature importance for each target
    print(f"\nCalculating feature importance for {len(target_names)} targets...")
    
    # Create the 2x1 subplot with taller proportion
    fig, axes = plt.subplots(1, 2, figsize=(20, 20))  # 20x20 size
    
    # Calculate and plot for each target
    for i, target_name in enumerate(target_names):
        print(f"\nAnalyzing feature importance for target: {target_name}")
        
        # Calculate importance values without plotting
        importance_values = compute_feature_importance(
            model=model,
            inputs=X_test_tensor,
            features=feature_names,
            method=method,
            baseline="random",
            baseline_samples=50,
            target_idx=i,
            target_name=target_name,
            skip_plot=True  # Prevent individual plotting
        )
        
        # Plot in the corresponding subplot
        axes[i].barh(feature_names[::-1], importance_values[::-1])  # Reverse order for better visualization5
        axes[i].set_xlabel('Feature Importance', fontsize=20, fontweight='bold')
        
        # Place title inside plot area with larger font
        title = f'Feature Importance for {target_name} ({method.upper()})'
        axes[i].text(0.5, 0.98, title,
                    horizontalalignment='center',
                    verticalalignment='top',
                    transform=axes[i].transAxes,
                    fontsize=20,
                    fontweight='bold')
        
        # Remove the original title
        axes[i].set_title('')
        
        # Add grid and style improvements
        axes[i].grid(True, linestyle='--', alpha=0.7)
        axes[i].spines['top'].set_visible(False)  # Remove top border
        
        # Remove y-axis labels from the right subplot
        if i == 1:  # right subplot
            axes[i].set_yticklabels([])
        else:  # left subplot
            # Increase y-axis labels font size and make bold
            axes[i].tick_params(axis='y', labelsize=20)
            for label in axes[i].get_yticklabels():
                label.set_fontweight('bold')
        
        # Increase x-axis labels font size and make bold
        axes[i].tick_params(axis='x', labelsize=20)
        for label in axes[i].get_xticklabels():
            label.set_fontweight('bold')
    
    # Adjust layout and save
    plt.tight_layout()
    plot_path = f'plots/feature_importance_{method}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to {plot_path}")
    plt.close()

if __name__ == "__main__":
    main() 