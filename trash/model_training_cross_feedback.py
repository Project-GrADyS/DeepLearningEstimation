"""
Dual Neural Network with Cross-Feedback for Formation Control

This module implements a dual neural network architecture with cross-feedback
for predicting communication range and failure rate in formation control.

Key Features:
- Two interdependent neural networks with cross-feedback
- Shared base features (size_formation, comm_delay, shape_error)
- Each network uses the prediction of the other as additional input
- Training with early stopping and learning rate scheduling
- Evaluation with MSE and R² metrics for each target
- Multiple iterations of cross-feedback during evaluation for stability

Architecture:
- NN_range: Predicts comm_range using base features + comm_failure prediction
- NN_failure: Predicts comm_failure using base features + comm_range prediction

Implementation Notes:
- During training, each network uses its own predictions for cross-feedback
- During evaluation, multiple iterations of cross-feedback are performed
- This approach prevents data leakage and ensures consistent evaluation

Dependencies:
    - numpy: Numerical computations
    - pandas: Data manipulation
    - torch: Deep learning framework
    - sklearn: Metrics and utilities
    - matplotlib: Visualization

Author: Laércio Lucchesi
Date: 2025-03-30
Version: 1.1
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
    def __init__(self, base_features: np.ndarray, comm_range: np.ndarray, comm_failure: np.ndarray):
        """
        Initialize the dataset.
        
        Args:
            base_features (np.ndarray): Base input features (size_formation, comm_delay, shape_error)
            comm_range (np.ndarray): Communication range values
            comm_failure (np.ndarray): Communication failure values
        """
        self.base_features = torch.FloatTensor(base_features)
        self.comm_range = torch.FloatTensor(comm_range)
        self.comm_failure = torch.FloatTensor(comm_failure)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.base_features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        return self.base_features[idx], self.comm_range[idx], self.comm_failure[idx]

class RangeNetwork(nn.Module):
    """
    Neural network for predicting communication range.
    """
    def __init__(self, input_size: int, hidden_sizes: List[int], dropout_rate: float = 0.0):
        """
        Initialize the range prediction network.
        
        Args:
            input_size (int): Number of input features (base features + comm_failure)
            hidden_sizes (List[int]): List of hidden layer sizes
            dropout_rate (float): Dropout rate for regularization
        """
        super(RangeNetwork, self).__init__()
        
        # Build the network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LeakyReLU(0.1))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer (single value for comm_range)
        layers.append(nn.Linear(prev_size, 1))
        
        # Combine all layers
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor (base features + comm_failure)
            
        Returns:
            torch.Tensor: Predicted communication range
        """
        return self.network(x)

class FailureNetwork(nn.Module):
    """
    Neural network for predicting communication failure rate.
    """
    def __init__(self, input_size: int, hidden_sizes: List[int], dropout_rate: float = 0.0):
        """
        Initialize the failure prediction network.
        
        Args:
            input_size (int): Number of input features (base features + comm_range)
            hidden_sizes (List[int]): List of hidden layer sizes
            dropout_rate (float): Dropout rate for regularization
        """
        super(FailureNetwork, self).__init__()
        
        # Build the network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LeakyReLU(0.1))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer (single value for comm_failure)
        layers.append(nn.Linear(prev_size, 1))
        
        # Combine all layers
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor (base features + comm_range)
            
        Returns:
            torch.Tensor: Predicted communication failure rate
        """
        return self.network(x)

def load_data(file_path: str, shape_error_type: str = 'avg_5min') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess the dataset.
    
    Args:
        file_path (str): Path to the dataset CSV file
        shape_error_type (str): Type of shape error to use ('avg_30s', 'avg_1min', 'avg_2min', 'avg_5min')
        
    Returns:
        Tuple[np.ndarray, ...]: 
            Training base features, training comm_range, training comm_failure,
            validation base features, validation comm_range, validation comm_failure,
            test base features, test comm_range, test comm_failure
    """
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Select base features and targets
    base_features = ['size_formation', 'comm_delay', shape_error_type]
    targets = ['comm_range', 'comm_failure']
    
    # Split data by dataset type
    train_data = data[data['dataset_type'] == 'training']
    val_data = data[data['dataset_type'] == 'validation']
    test_data = data[data['dataset_type'] == 'test']
    
    # Extract base features and targets
    X_train_base = train_data[base_features].values
    y_train_range = train_data[targets[0]].values.reshape(-1, 1)
    y_train_failure = train_data[targets[1]].values.reshape(-1, 1)
    
    X_val_base = val_data[base_features].values
    y_val_range = val_data[targets[0]].values.reshape(-1, 1)
    y_val_failure = val_data[targets[1]].values.reshape(-1, 1)
    
    X_test_base = test_data[base_features].values
    y_test_range = test_data[targets[0]].values.reshape(-1, 1)
    y_test_failure = test_data[targets[1]].values.reshape(-1, 1)
    
    return X_train_base, y_train_range, y_train_failure, X_val_base, y_val_range, y_val_failure, X_test_base, y_test_range, y_test_failure

def train_networks(
    range_net: nn.Module,
    failure_net: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    range_optimizer: optim.Optimizer,
    failure_optimizer: optim.Optimizer,
    range_scheduler: ReduceLROnPlateau,
    failure_scheduler: ReduceLROnPlateau,
    num_epochs: int,
    patience: int,
    device: torch.device
) -> Tuple[List[float], List[float], List[float], List[float], int]:
    """
    Train both networks with cross-feedback and early stopping.
    
    Args:
        range_net (nn.Module): The range prediction network
        failure_net (nn.Module): The failure prediction network
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        criterion (nn.Module): Loss function
        range_optimizer (optim.Optimizer): Optimizer for range network
        failure_optimizer (optim.Optimizer): Optimizer for failure network
        range_scheduler (ReduceLROnPlateau): Learning rate scheduler for range network
        failure_scheduler (ReduceLROnPlateau): Learning rate scheduler for failure network
        num_epochs (int): Maximum number of epochs
        patience (int): Early stopping patience
        device (torch.device): Device to use for training
        
    Returns:
        Tuple[List[float], List[float], List[float], List[float], int]: 
            Training losses for range, training losses for failure,
            validation losses for range, validation losses for failure,
            best epoch
    """
    range_net = range_net.to(device)
    failure_net = failure_net.to(device)
    
    range_train_losses = []
    failure_train_losses = []
    range_val_losses = []
    failure_val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        range_net.train()
        failure_net.train()
        range_train_loss = 0.0
        failure_train_loss = 0.0
        
        for base_features, comm_range, comm_failure in train_loader:
            base_features = base_features.to(device)
            comm_range = comm_range.to(device)
            comm_failure = comm_failure.to(device)
            
            # Initialize predictions for this batch
            pred_failure = torch.zeros_like(comm_failure).to(device)
            pred_range = torch.zeros_like(comm_range).to(device)
            
            # Perform multiple iterations of cross-feedback
            for _ in range(3):  # 3 iterations of cross-feedback
                # Forward pass for range network
                range_input = torch.cat([base_features, pred_failure], dim=1)
                range_output = range_net(range_input)
                
                # Forward pass for failure network
                failure_input = torch.cat([base_features, range_output.detach()], dim=1)
                failure_output = failure_net(failure_input)
                
                # Update predictions for next iteration
                pred_range = range_output.detach()
                pred_failure = failure_output.detach()
            
            # Calculate losses using final predictions
            range_loss = criterion(range_output, comm_range)
            failure_loss = criterion(failure_output, comm_failure)
            
            # Backward pass and optimize
            range_optimizer.zero_grad()
            range_loss.backward()
            range_optimizer.step()
            
            failure_optimizer.zero_grad()
            failure_loss.backward()
            failure_optimizer.step()
            
            range_train_loss += range_loss.item()
            failure_train_loss += failure_loss.item()
        
        range_train_loss /= len(train_loader)
        failure_train_loss /= len(train_loader)
        range_train_losses.append(range_train_loss)
        failure_train_losses.append(failure_train_loss)
        
        # Validation phase
        range_net.eval()
        failure_net.eval()
        range_val_loss = 0.0
        failure_val_loss = 0.0
        
        with torch.no_grad():
            for base_features, comm_range, comm_failure in val_loader:
                base_features = base_features.to(device)
                comm_range = comm_range.to(device)
                comm_failure = comm_failure.to(device)
                
                # Initialize predictions for this batch
                pred_failure = torch.zeros_like(comm_failure).to(device)
                pred_range = torch.zeros_like(comm_range).to(device)
                
                # Perform multiple iterations of cross-feedback
                for _ in range(3):  # 3 iterations of cross-feedback
                    # Forward pass for range network
                    range_input = torch.cat([base_features, pred_failure], dim=1)
                    range_output = range_net(range_input)
                    
                    # Forward pass for failure network
                    failure_input = torch.cat([base_features, range_output], dim=1)
                    failure_output = failure_net(failure_input)
                    
                    # Update predictions for next iteration
                    pred_range = range_output
                    pred_failure = failure_output
                
                # Calculate losses using final predictions
                range_loss = criterion(range_output, comm_range)
                failure_loss = criterion(failure_output, comm_failure)
                
                range_val_loss += range_loss.item()
                failure_val_loss += failure_loss.item()
        
        range_val_loss /= len(val_loader)
        failure_val_loss /= len(val_loader)
        range_val_losses.append(range_val_loss)
        failure_val_losses.append(failure_val_loss)
        
        # Calculate total validation loss
        total_val_loss = range_val_loss + failure_val_loss
        
        # Update learning rate schedulers
        range_scheduler.step(range_val_loss)
        failure_scheduler.step(failure_val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Range - Train Loss: {range_train_loss:.4f}, Val Loss: {range_val_loss:.4f}, LR: {range_optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Failure - Train Loss: {failure_train_loss:.4f}, Val Loss: {failure_val_loss:.4f}, LR: {failure_optimizer.param_groups[0]['lr']:.6f}")
        
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

def evaluate_networks(
    range_net: nn.Module,
    failure_net: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_iterations: int = 3
) -> Tuple[float, float, Dict[str, float]]:
    """
    Evaluate both networks on the test set with multiple iterations of cross-feedback.
    
    Args:
        range_net (nn.Module): The range prediction network
        failure_net (nn.Module): The failure prediction network
        test_loader (DataLoader): DataLoader for test data
        criterion (nn.Module): Loss function
        device (torch.device): Device to use for evaluation
        num_iterations (int): Number of cross-feedback iterations to perform
        
    Returns:
        Tuple[float, float, Dict[str, float]]: 
            Test loss for range, test loss for failure, metrics dictionary
    """
    range_net.eval()
    failure_net.eval()
    range_test_loss = 0.0
    failure_test_loss = 0.0
    all_range_preds = []
    all_failure_preds = []
    all_range_targets = []
    all_failure_targets = []
    
    with torch.no_grad():
        for base_features, comm_range, comm_failure in test_loader:
            base_features = base_features.to(device)
            comm_range = comm_range.to(device)
            comm_failure = comm_failure.to(device)
            
            # Initialize predictions for this batch
            pred_failure = torch.zeros_like(comm_failure).to(device)
            pred_range = torch.zeros_like(comm_range).to(device)
            
            # Perform multiple iterations of cross-feedback
            for _ in range(num_iterations):
                # Forward pass for range network
                range_input = torch.cat([base_features, pred_failure], dim=1)
                range_output = range_net(range_input)
                
                # Forward pass for failure network
                failure_input = torch.cat([base_features, range_output], dim=1)
                failure_output = failure_net(failure_input)
                
                # Update predictions for next iteration
                pred_range = range_output
                pred_failure = failure_output
            
            # Calculate losses using final predictions
            range_loss = criterion(range_output, comm_range)
            failure_loss = criterion(failure_output, comm_failure)
            
            range_test_loss += range_loss.item()
            failure_test_loss += failure_loss.item()
            
            all_range_preds.append(range_output.cpu().numpy())
            all_failure_preds.append(failure_output.cpu().numpy())
            all_range_targets.append(comm_range.cpu().numpy())
            all_failure_targets.append(comm_failure.cpu().numpy())
    
    range_test_loss /= len(test_loader)
    failure_test_loss /= len(test_loader)
    
    # Convert predictions and targets to numpy arrays
    all_range_preds = np.vstack(all_range_preds)
    all_failure_preds = np.vstack(all_failure_preds)
    all_range_targets = np.vstack(all_range_targets)
    all_failure_targets = np.vstack(all_failure_targets)
    
    # Calculate metrics for each target
    range_mse = mean_squared_error(all_range_targets, all_range_preds)
    range_r2 = r2_score(all_range_targets, all_range_preds)
    
    failure_mse = mean_squared_error(all_failure_targets, all_failure_preds)
    failure_r2 = r2_score(all_failure_targets, all_failure_preds)
    
    metrics = {
        'comm_range_mse': range_mse,
        'comm_range_r2': range_r2,
        'comm_failure_mse': failure_mse,
        'comm_failure_r2': failure_r2
    }
    
    return range_test_loss, failure_test_loss, metrics

def plot_losses(
    range_train_losses: List[float], 
    failure_train_losses: List[float], 
    range_val_losses: List[float], 
    failure_val_losses: List[float], 
    save_path: Optional[str] = None
):
    """
    Plot training and validation losses for both networks.
    
    Args:
        range_train_losses (List[float]): Training losses for range network
        failure_train_losses (List[float]): Training losses for failure network
        range_val_losses (List[float]): Validation losses for range network
        failure_val_losses (List[float]): Validation losses for failure network
        save_path (Optional[str]): Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot for range network
    plt.subplot(1, 2, 1)
    plt.plot(range_train_losses, label='Training Loss')
    plt.plot(range_val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Range Network Losses')
    plt.legend()
    plt.grid(True)
    
    # Plot for failure network
    plt.subplot(1, 2, 2)
    plt.plot(failure_train_losses, label='Training Loss')
    plt.plot(failure_val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Failure Network Losses')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Loss plots saved to {save_path}")
    
    plt.show()

def main():
    """
    Main function to execute the dual network training pipeline.
    """
    # Configuration
    DATASET_PATH = 'dataset.csv'
    HIDDEN_SIZES = [128, 64, 32]  # Hidden layer sizes
    DROPOUT_RATE = 0.1  # Dropout rate
    BATCH_SIZE = 64  # Batch size
    NUM_EPOCHS = 1000  # Number of epochs
    PATIENCE = 50  # Early stopping patience
    LEARNING_RATE = 0.001  # Learning rate
    WEIGHT_DECAY = 0.0001  # Weight decay
    
    # Ask user to choose shape error type
    print("Choose the shape error metric to use:")
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
    if choice == "1":
        SHAPE_ERROR_TYPE = 'avg_30s'
        print("Using avg_30s shape error")
    elif choice == "2":
        SHAPE_ERROR_TYPE = 'avg_1min'
        print("Using avg_1min shape error")
    elif choice == "3":
        SHAPE_ERROR_TYPE = 'avg_2min'
        print("Using avg_2min shape error")
    elif choice == "4":
        SHAPE_ERROR_TYPE = 'avg_5min'
        print("Using avg_5min shape error")
    
    # Create output directory
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading data from {DATASET_PATH}...")
    X_train_base, y_train_range, y_train_failure, X_val_base, y_val_range, y_val_failure, X_test_base, y_test_range, y_test_failure = load_data(DATASET_PATH, SHAPE_ERROR_TYPE)
    
    # Create datasets
    train_dataset = FormationDataset(X_train_base, y_train_range, y_train_failure)
    val_dataset = FormationDataset(X_val_base, y_val_range, y_val_failure)
    test_dataset = FormationDataset(X_test_base, y_test_range, y_test_failure)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Create networks
    base_input_size = X_train_base.shape[1]  # Number of base features
    
    # Range network: base features + comm_failure
    range_input_size = base_input_size + 1
    range_net = RangeNetwork(range_input_size, HIDDEN_SIZES, DROPOUT_RATE)
    
    # Failure network: base features + comm_range
    failure_input_size = base_input_size + 1
    failure_net = FailureNetwork(failure_input_size, HIDDEN_SIZES, DROPOUT_RATE)
    
    # Define loss function and optimizers
    criterion = nn.SmoothL1Loss()  # Using SmoothL1Loss instead of MSELoss
    range_optimizer = optim.AdamW(range_net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    failure_optimizer = optim.AdamW(failure_net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Define learning rate schedulers
    range_scheduler = ReduceLROnPlateau(range_optimizer, mode='min', factor=0.5, patience=10)
    failure_scheduler = ReduceLROnPlateau(failure_optimizer, mode='min', factor=0.5, patience=10)
    
    # Train networks
    print("Starting network training...")
    start_time = time.time()
    range_train_losses, failure_train_losses, range_val_losses, failure_val_losses, best_epoch = train_networks(
        range_net, failure_net, train_loader, val_loader, criterion, 
        range_optimizer, failure_optimizer, range_scheduler, failure_scheduler, 
        NUM_EPOCHS, PATIENCE, device
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Plot losses
    plot_losses(
        range_train_losses, failure_train_losses, 
        range_val_losses, failure_val_losses, 
        f'plots/cross_feedback_loss_plot_{SHAPE_ERROR_TYPE}.png'
    )
    
    # Evaluate networks
    print("\nEvaluating networks on test set...")
    range_test_loss, failure_test_loss, metrics = evaluate_networks(
        range_net, failure_net, test_loader, criterion, device, num_iterations=5
    )
    
    # Print test results
    print("\nTest Results:")
    print(f"Range Network Test Loss: {range_test_loss:.6f}")
    print(f"Failure Network Test Loss: {failure_test_loss:.6f}")
    print(f"Total Test Loss: {range_test_loss + failure_test_loss:.6f}")
    
    print("\nMetrics for each target variable:")
    print("comm_range:")
    print(f"  MSE: {metrics['comm_range_mse']:.6f}")
    print(f"  R² Score: {metrics['comm_range_r2']:.6f}")
    print("comm_failure:")
    print(f"  MSE: {metrics['comm_failure_mse']:.6f}")
    print(f"  R² Score: {metrics['comm_failure_r2']:.6f}")
    
    # Save models
    model_path = f'models/cross_feedback_model_{SHAPE_ERROR_TYPE}.pth'
    torch.save({
        'range_net_state_dict': range_net.state_dict(),
        'failure_net_state_dict': failure_net.state_dict(),
        'range_optimizer_state_dict': range_optimizer.state_dict(),
        'failure_optimizer_state_dict': failure_optimizer.state_dict(),
        'range_scheduler_state_dict': range_scheduler.state_dict(),
        'failure_scheduler_state_dict': failure_scheduler.state_dict(),
        'range_train_losses': range_train_losses,
        'failure_train_losses': failure_train_losses,
        'range_val_losses': range_val_losses,
        'failure_val_losses': failure_val_losses,
        'best_epoch': best_epoch,
        'range_test_loss': range_test_loss,
        'failure_test_loss': failure_test_loss,
        'metrics': metrics,
        'config': {
            'base_input_size': base_input_size,
            'range_input_size': range_input_size,
            'failure_input_size': failure_input_size,
            'hidden_sizes': HIDDEN_SIZES,
            'dropout_rate': DROPOUT_RATE,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'shape_error_type': SHAPE_ERROR_TYPE
        }
    }, model_path)
    print(f"\nModels saved to {model_path}")

if __name__ == "__main__":
    main() 