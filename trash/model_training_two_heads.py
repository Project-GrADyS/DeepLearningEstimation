"""
Multi-Head Neural Network for Regression

This module implements a neural network with a shared trunk and two separate heads
for predicting two continuous variables: comm_range and comm_failure.

Key Features:
- Shared trunk network with common features
- Two separate heads for predicting different targets
- Training pipeline with early stopping
- Model evaluation and metrics calculation
- Model saving and loading
- User selection of shape error type

Architecture:
- Shared trunk: Processes common features (size_formation, comm_delay, shape_error)
- Head 1: Predicts comm_range
- Head 2: Predicts comm_failure

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

class MultiHeadFormationModel(nn.Module):
    """
    Multi-head neural network model for formation control prediction.
    
    This model consists of a shared trunk network and two separate heads
    for predicting comm_range and comm_failure.
    """
    def __init__(self, input_size: int, hidden_sizes: List[int], dropout_rate: float = 0.0):
        """
        Initialize the multi-head model.
        
        Args:
            input_size (int): Number of input features
            hidden_sizes (List[int]): List of hidden layer sizes for the shared trunk
            dropout_rate (float): Dropout rate for regularization
        """
        super(MultiHeadFormationModel, self).__init__()
        
        # Build the shared trunk network
        trunk_layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            trunk_layers.append(nn.Linear(prev_size, hidden_size))
            trunk_layers.append(nn.LeakyReLU(0.1))
            if dropout_rate > 0:
                trunk_layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        self.shared_trunk = nn.Sequential(*trunk_layers)
        
        # Build the range head (outputs comm_range)
        self.range_head = nn.Sequential(
            nn.Linear(prev_size, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1)
        )
        
        # Build the failure head (outputs comm_failure)
        self.failure_head = nn.Sequential(
            nn.Linear(prev_size, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensors for comm_range and comm_failure
        """
        # Pass through shared trunk
        shared_features = self.shared_trunk(x)
        
        # Pass through respective heads
        range_output = self.range_head(shared_features)
        failure_output = self.failure_head(shared_features)
        
        return range_output, failure_output

def get_shape_error_type() -> str:
    """
    Ask the user to select the shape error type.
    
    Returns:
        str: Selected shape error type
    """
    print("\nChoose the shape error metric to use:")
    print("1. avg_30s (Average shape error for first 30 seconds)")
    print("2. avg_1min (Average shape error for first 1 minute)")
    print("3. avg_2min (Average shape error for first 2 minutes)")
    print("4. avg_5min (Average shape error for first 5 minutes)")
    
    while True:
        choice = input("Enter option number (1-4): ")
        if choice in ["1", "2", "3", "4"]:
            break
        else:
            print("Invalid input. Please enter 1, 2, 3, or 4.")
    
    # Set shape error type based on user choice
    shape_error_type = {
        "1": "avg_30s",
        "2": "avg_1min",
        "3": "avg_2min",
        "4": "avg_5min"
    }[choice]
    
    print(f"\nSelected shape error type: {shape_error_type}")
    return shape_error_type

def load_data(file_path: str, shape_error_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess the dataset.
    
    Args:
        file_path (str): Path to the dataset CSV file
        shape_error_type (str): Type of shape error to use
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            Training features, training targets, validation features, validation targets, 
            test features, test targets
    """
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Select features and targets
    features = ['size_formation', 'comm_delay', shape_error_type]
    targets = ['comm_range', 'comm_failure']
    
    print(f"\nFeatures: {features}")
    print(f"Targets: {targets}")
    
    # Split data by dataset type
    train_data = data[data['dataset_type'] == 'training']
    val_data = data[data['dataset_type'] == 'validation']
    test_data = data[data['dataset_type'] == 'test']
    
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
) -> Tuple[List[float], List[float], List[float], List[float], int]:
    """
    Train the multi-head model with early stopping.
    
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
        Tuple[List[float], List[float], List[float], List[float], int]: 
            Training losses for range, training losses for failure,
            validation losses for range, validation losses for failure,
            best epoch
    """
    model = model.to(device)
    
    range_train_losses = []
    failure_train_losses = []
    range_val_losses = []
    failure_val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        range_train_loss = 0.0
        failure_train_loss = 0.0
        
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            # Split targets into range and failure
            range_target = targets[:, 0].view(-1, 1)
            failure_target = targets[:, 1].view(-1, 1)
            
            # Forward pass
            range_output, failure_output = model(features)
            
            # Calculate losses
            range_loss = criterion(range_output, range_target)
            failure_loss = criterion(failure_output, failure_target)
            
            # Total loss is the sum of both losses
            total_loss = range_loss + failure_loss
            
            # Backward pass and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            range_train_loss += range_loss.item()
            failure_train_loss += failure_loss.item()
        
        range_train_loss /= len(train_loader)
        failure_train_loss /= len(train_loader)
        range_train_losses.append(range_train_loss)
        failure_train_losses.append(failure_train_loss)
        
        # Validation phase
        model.eval()
        range_val_loss = 0.0
        failure_val_loss = 0.0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                
                # Split targets into range and failure
                range_target = targets[:, 0].view(-1, 1)
                failure_target = targets[:, 1].view(-1, 1)
                
                # Forward pass
                range_output, failure_output = model(features)
                
                # Calculate losses
                range_loss = criterion(range_output, range_target)
                failure_loss = criterion(failure_output, failure_target)
                
                range_val_loss += range_loss.item()
                failure_val_loss += failure_loss.item()
        
        range_val_loss /= len(val_loader)
        failure_val_loss /= len(val_loader)
        range_val_losses.append(range_val_loss)
        failure_val_losses.append(failure_val_loss)
        
        # Total validation loss is the sum of both losses
        total_val_loss = range_val_loss + failure_val_loss
        
        # Update learning rate scheduler
        scheduler.step(total_val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Range Train Loss: {range_train_loss:.4f}, Range Val Loss: {range_val_loss:.4f}, "
                  f"Failure Train Loss: {failure_train_loss:.4f}, Failure Val Loss: {failure_val_loss:.4f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_epoch = epoch
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return range_train_losses, failure_train_losses, range_val_losses, failure_val_losses, best_epoch

def evaluate_model(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate the model on the test set.
    
    Args:
        model (nn.Module): The trained model
        test_loader (DataLoader): DataLoader for test data
        criterion (nn.Module): Loss function
        device (torch.device): Device to use for evaluation
        
    Returns:
        Tuple[float, Dict[str, float]]: Test loss and metrics dictionary
    """
    model.eval()
    test_loss = 0.0
    all_range_preds = []
    all_failure_preds = []
    all_range_targets = []
    all_failure_targets = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            
            # Split targets into range and failure
            range_target = targets[:, 0].view(-1, 1)
            failure_target = targets[:, 1].view(-1, 1)
            
            # Forward pass
            range_output, failure_output = model(features)
            
            # Calculate losses
            range_loss = criterion(range_output, range_target)
            failure_loss = criterion(failure_output, failure_target)
            
            # Total loss is the sum of both losses
            total_loss = range_loss + failure_loss
            test_loss += total_loss.item()
            
            # Store predictions and targets
            all_range_preds.append(range_output.cpu().numpy())
            all_failure_preds.append(failure_output.cpu().numpy())
            all_range_targets.append(range_target.cpu().numpy())
            all_failure_targets.append(failure_target.cpu().numpy())
    
    test_loss /= len(test_loader)
    
    # Convert predictions and targets to numpy arrays
    all_range_preds = np.vstack(all_range_preds)
    all_failure_preds = np.vstack(all_failure_preds)
    all_range_targets = np.vstack(all_range_targets)
    all_failure_targets = np.vstack(all_failure_targets)
    
    # Calculate metrics for each target
    metrics = {}
    
    # Metrics for comm_range
    range_mse = mean_squared_error(all_range_targets, all_range_preds)
    range_r2 = r2_score(all_range_targets, all_range_preds)
    metrics['comm_range_mse'] = range_mse
    metrics['comm_range_r2'] = range_r2
    
    # Metrics for comm_failure
    failure_mse = mean_squared_error(all_failure_targets, all_failure_preds)
    failure_r2 = r2_score(all_failure_targets, all_failure_preds)
    metrics['comm_failure_mse'] = failure_mse
    metrics['comm_failure_r2'] = failure_r2
    
    return test_loss, metrics

def plot_losses(range_train_losses, failure_train_losses, range_val_losses, failure_val_losses, save_path=None):
    """
    Plot training and validation losses for both heads.
    
    Args:
        range_train_losses (List[float]): Training losses for comm_range
        failure_train_losses (List[float]): Training losses for comm_failure
        range_val_losses (List[float]): Validation losses for comm_range
        failure_val_losses (List[float]): Validation losses for comm_failure
        save_path (Optional[str]): Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot for comm_range
    plt.subplot(1, 2, 1)
    plt.plot(range_train_losses, label='Training Loss')
    plt.plot(range_val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss for comm_range')
    plt.legend()
    plt.grid(True)
    
    # Plot for comm_failure
    plt.subplot(1, 2, 2)
    plt.plot(failure_train_losses, label='Training Loss')
    plt.plot(failure_val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss for comm_failure')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
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
    HIDDEN_SIZES = [128, 64, 32]  # Hidden layer sizes for shared trunk
    DROPOUT_RATE = 0.1  # Dropout rate
    BATCH_SIZE = 64  # Batch size
    NUM_EPOCHS = 1000  # Number of epochs
    PATIENCE = 50  # Early stopping patience
    LEARNING_RATE = 0.001  # Learning rate
    WEIGHT_DECAY = 0.0001  # Weight decay
    
    # Get shape error type from user
    shape_error_type = get_shape_error_type()
    
    # Create output directory
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading data from {DATASET_PATH}...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(DATASET_PATH, shape_error_type)
    
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
    model = MultiHeadFormationModel(input_size, HIDDEN_SIZES, DROPOUT_RATE)
    
    # Define loss function and optimizer
    criterion = nn.SmoothL1Loss()  # Using SmoothL1Loss instead of MSELoss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Define learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Train model
    print("Starting model training...")
    start_time = time.time()
    range_train_losses, failure_train_losses, range_val_losses, failure_val_losses, best_epoch = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS, PATIENCE, device
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Plot losses
    plot_losses(
        range_train_losses, failure_train_losses, 
        range_val_losses, failure_val_losses, 
        f'plots/loss_plot_two_heads_{shape_error_type}.png'
    )
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    test_loss, metrics = evaluate_model(model, test_loader, criterion, device)
    
    # Print test results
    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.6f}")
    print("\nMetrics for comm_range:")
    print(f"  MSE: {metrics['comm_range_mse']:.6f}")
    print(f"  R² Score: {metrics['comm_range_r2']:.6f}")
    print("\nMetrics for comm_failure:")
    print(f"  MSE: {metrics['comm_failure_mse']:.6f}")
    print(f"  R² Score: {metrics['comm_failure_r2']:.6f}")
    
    # Save model
    model_path = f'models/formation_model_two_heads_{shape_error_type}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'range_train_losses': range_train_losses,
        'failure_train_losses': failure_train_losses,
        'range_val_losses': range_val_losses,
        'failure_val_losses': failure_val_losses,
        'best_epoch': best_epoch,
        'test_loss': test_loss,
        'metrics': metrics,
        'config': {
            'input_size': input_size,
            'hidden_sizes': HIDDEN_SIZES,
            'dropout_rate': DROPOUT_RATE,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'shape_error_type': shape_error_type
        }
    }, model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    main() 