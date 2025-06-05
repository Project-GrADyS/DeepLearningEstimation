"""
Model Training Module for Distributed Formation Control (Version 2)

This module is responsible for training a deep neural network model to predict
communication parameters based on formation characteristics and time series data.

Key Features:
- Data loading from NPZ format
- Temporal summary layer for time series data
- Model architecture with input layer
- Training pipeline with early stopping
- Model evaluation and metrics calculation
- Model saving and loading

Dependencies:
    - numpy: Numerical computations
    - torch: Deep learning framework
    - sklearn: Metrics and utilities
    - matplotlib: Visualization
    - tqdm: Progress bar

Author: Laércio Lucchesi
Date: 2025-04-20
Version: 2.0
"""

import numpy as np
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
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Dataset type constants
TRAIN, VAL, TEST = 0, 1, 2

# --- TemporalSummary ---
class TemporalSummary(nn.Module):
    """
    A module that summarizes time series data using different window sizes.
    
    This module takes a time series input and produces a fixed number of outputs
    by applying different window sizes to the input and computing the mean.
    """
    def __init__(self, time_steps: int = 5858, num_outputs: int = 10, mode: str = 'linear', min_ratio: float = 0.1):
        """
        Initialize the TemporalSummary module.
        
        Args:
            time_steps (int): Number of time steps in the input data
            num_outputs (int): Number of outputs to produce
            mode (str): Mode for determining window sizes ('short', 'linear', 'long')
            min_ratio (float): Minimum ratio for window sizes
        """
        super().__init__()
        self.time_steps = time_steps
        self.num_outputs = num_outputs
        self.mode = mode.lower()
        self.min_ratio = min_ratio

        # Special case for num_outputs = 1
        if num_outputs == 1:
            self.window_sizes = [time_steps]  # Use all time steps
            return

        x = np.arange(1, num_outputs + 1)

        if self.mode == 'short':
            alpha_base = 1 / x
        elif self.mode == 'linear':
            alpha_base = 1 - (x - 1) / num_outputs
        elif self.mode == 'long':
            alpha_base = 1 + 1 / num_outputs - 1 / (num_outputs - x + 1)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        alpha_norm = (alpha_base - alpha_base.min()) / (alpha_base.max() - alpha_base.min())
        alpha_scaled = self.min_ratio + (1.0 - self.min_ratio) * alpha_norm

        self.window_sizes = [max(1, int(round(time_steps * a))) for a in alpha_scaled]

    def forward(self, x):
        """
        Forward pass through the module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time_steps)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels * num_outputs)
        """
        outputs = []
        for window_size in self.window_sizes:
            slice_ = x[:, :, -window_size:]
            avg = slice_.mean(dim=2)
            outputs.append(avg)
        return torch.cat(outputs, dim=1)

# --- InputLayer ---
class InputLayer(nn.Module):
    """
    Input layer that processes static features and time series data.
    
    This layer combines static features with summarized time series data
    from multiple sources.
    """
    def __init__(self, time_steps, num_outputs, mode='linear', min_ratio=0.1):
        """
        Initialize the InputLayer module.
        
        Args:
            time_steps (int): Number of time steps in the time series data
            num_outputs (int): Number of outputs for each TemporalSummary
            mode (str): Mode for determining window sizes
            min_ratio (float): Minimum ratio for window sizes
        """
        super().__init__()
        self.shape_summary = TemporalSummary(time_steps, num_outputs, mode, min_ratio)
        self.presence_summary = TemporalSummary(time_steps, num_outputs, mode, min_ratio)
        self.leader_summary = TemporalSummary(time_steps, num_outputs, mode, min_ratio)

    def forward(self, x_static, x_shape, x_presence, x_leader):
        """
        Forward pass through the module.
        
        Args:
            x_static (torch.Tensor): Static features (size_formation, comm_delay)
            x_shape (torch.Tensor): Shape error time series
            x_presence (torch.Tensor): Presence error time series
            x_leader (torch.Tensor): Leader offset time series
            
        Returns:
            torch.Tensor: Combined features
        """
        s1 = self.shape_summary(x_shape)
        s2 = self.presence_summary(x_presence)
        s3 = self.leader_summary(x_leader)
        return torch.cat([x_static, s1, s2, s3], dim=1)

class FormationDataset(Dataset):
    """
    Custom Dataset class for formation control data.
    """
    def __init__(self, static_features: np.ndarray, shape_error: np.ndarray, 
                 presence_error: np.ndarray, leader_off: np.ndarray, targets: np.ndarray):
        """
        Initialize the dataset.
        
        Args:
            static_features (np.ndarray): Static features (size_formation, comm_delay)
            shape_error (np.ndarray): Shape error time series
            presence_error (np.ndarray): Presence error time series
            leader_off (np.ndarray): Leader offset time series
            targets (np.ndarray): Target values (comm_range, comm_failure)
        """
        self.static_features = torch.FloatTensor(static_features)
        self.shape_error = torch.FloatTensor(shape_error)
        self.presence_error = torch.FloatTensor(presence_error)
        self.leader_off = torch.FloatTensor(leader_off)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.static_features)
    
    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """Get a sample from the dataset."""
        return (self.static_features[idx], 
                self.shape_error[idx].unsqueeze(0), 
                self.presence_error[idx].unsqueeze(0), 
                self.leader_off[idx].unsqueeze(0)), self.targets[idx]

class FormationModel(nn.Module):
    """
    Deep neural network model for formation control prediction.
    """
    def __init__(self, input_layer: InputLayer, hidden_sizes: List[int], output_size: int, dropout_rate: float = 0.0, include_size_formation: bool = True):
        """
        Initialize the model.
        
        Args:
            input_layer (InputLayer): Input layer for processing features
            hidden_sizes (List[int]): List of hidden layer sizes
            output_size (int): Number of output targets
            dropout_rate (float): Dropout rate for regularization
            include_size_formation (bool): Whether to include size_formation feature
        """
        super(FormationModel, self).__init__()
        
        self.input_layer = input_layer
        
        # Calculate input size for the network after the input layer
        # Static features (1 or 2) + 3 time series with num_outputs each
        num_static_features = 2 if include_size_formation else 1
        input_size = num_static_features + 3 * input_layer.shape_summary.num_outputs
        
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
    
    def forward(self, x_static, x_shape, x_presence, x_leader):
        """
        Forward pass through the network.
        
        Args:
            x_static (torch.Tensor): Static features
            x_shape (torch.Tensor): Shape error time series
            x_presence (torch.Tensor): Presence error time series
            x_leader (torch.Tensor): Leader offset time series
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Process inputs through the input layer
        x = self.input_layer(x_static, x_shape, x_presence, x_leader)
        
        # Pass through the network
        return self.network(x)

def get_temporal_summary_params() -> Tuple[int, str, float]:
    """
    Ask the user to define the parameters for the TemporalSummary layer.
    
    Returns:
        Tuple[int, str, float]: Number of outputs, mode, and minimum ratio
    """
    # Ask for number of outputs
    print("\nDefine the number of outputs for the TemporalSummary layer:")
    print("This determines how many summary features will be extracted from each time series.")
    
    while True:
        try:
            num_outputs = int(input("Enter number of outputs (1-100): "))
            if 1 <= num_outputs <= 100:
                break
            else:
                print("Invalid input. Please enter a number between 1 and 100.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Ask for mode
    print("\nChoose the mode for determining window sizes:")
    print("1. short: Emphasizes recent data")
    print("2. linear: Equal emphasis across time")
    print("3. long: Emphasizes older data")
    
    while True:
        mode_choice = input("Enter option number (1-3): ")
        if mode_choice in ["1", "2", "3"]:
            break
        else:
            print("Invalid input. Please enter 1, 2, or 3.")
    
    mode_map = {
        "1": "short",
        "2": "linear",
        "3": "long"
    }
    mode = mode_map[mode_choice]
    
    # Ask for minimum ratio
    print("\nDefine the minimum ratio for window sizes:")
    print("This determines the smallest window size relative to the total time steps.")
    
    while True:
        try:
            min_ratio = float(input("Enter minimum ratio (0.1-0.5): "))
            if 0.1 <= min_ratio <= 0.5:
                break
            else:
                print("Invalid input. Please enter a number between 0.1 and 0.5.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    return num_outputs, mode, min_ratio

def split_dataset_traditional(static_features: np.ndarray, shape_error: np.ndarray, 
                            presence_error: np.ndarray, leader_off: np.ndarray, 
                            targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the dataset using traditional 80/10/10 split.
    
    Args:
        static_features (np.ndarray): Static features array
        shape_error (np.ndarray): Shape error array
        presence_error (np.ndarray): Presence error array
        leader_off (np.ndarray): Leader offset array
        targets (np.ndarray): Targets array
        
    Returns:
        Tuple containing the same elements as load_data function
    """
    # First split: 80% train, 20% temp
    train_idx, temp_idx = train_test_split(
        np.arange(len(static_features)), 
        train_size=0.8, 
        random_state=42
    )
    
    # Second split: split temp into 50% val, 50% test (which gives us 10% each of total)
    val_idx, test_idx = train_test_split(
        temp_idx, 
        train_size=0.5, 
        random_state=42
    )
    
    # Split all data according to indices
    X_train_static = static_features[train_idx]
    X_train_shape = shape_error[train_idx]
    X_train_presence = presence_error[train_idx]
    X_train_leader = leader_off[train_idx]
    y_train = targets[train_idx]
    
    X_val_static = static_features[val_idx]
    X_val_shape = shape_error[val_idx]
    X_val_presence = presence_error[val_idx]
    X_val_leader = leader_off[val_idx]
    y_val = targets[val_idx]
    
    X_test_static = static_features[test_idx]
    X_test_shape = shape_error[test_idx]
    X_test_presence = presence_error[test_idx]
    X_test_leader = leader_off[test_idx]
    y_test = targets[test_idx]
    
    return (X_train_static, X_train_shape, X_train_presence, X_train_leader, y_train,
            X_val_static, X_val_shape, X_val_presence, X_val_leader, y_val,
            X_test_static, X_test_shape, X_test_presence, X_test_leader, y_test)

def get_split_type() -> Tuple[str, Optional[int]]:
    """
    Ask the user to choose between split types and K value if needed.
    
    Returns:
        Tuple[str, Optional[int]]: Split type ('a_priori', 'traditional', or 'k_fold') and K value if k_fold
    """
    print("\nChoose the dataset split type:")
    print("1. A priori split (using dataset_type field)")
    print("2. Traditional split (80% train, 10% validation, 10% test)")
    print("3. K-fold cross-validation with traditional split")
    
    while True:
        choice = input("Enter option number (1-3): ")
        if choice in ["1", "2", "3"]:
            break
        else:
            print("Invalid input. Please enter 1, 2, or 3.")
    
    if choice == "1":
        return "a_priori", None
    elif choice == "2":
        return "traditional", None
    else:  # choice == "3"
        while True:
            try:
                k = int(input("Enter the number of folds (3-10): "))
                if 3 <= k <= 10:
                    break
                else:
                    print("Invalid input. Please enter a number between 3 and 10.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        return "k_fold", k

def get_feature_choice() -> bool:
    """
    Ask the user if they want to include the size_formation feature.
    
    Returns:
        bool: True if size_formation should be included, False otherwise
    """
    print("\nDo you want to include the size_formation feature in the model?")
    print("1. Yes")
    print("2. No")
    
    while True:
        choice = input("Enter option number (1-2): ")
        if choice in ["1", "2"]:
            break
        else:
            print("Invalid input. Please enter 1 or 2.")
    
    return choice == "1"

def train_with_k_fold(model: nn.Module, 
                     static_features: np.ndarray,
                     shape_error: np.ndarray,
                     presence_error: np.ndarray,
                     leader_off: np.ndarray,
                     targets: np.ndarray,
                     k: int,
                     time_steps: int,
                     num_outputs: int,
                     mode: str,
                     min_ratio: float,
                     hidden_sizes: List[int],
                     output_size: int,
                     dropout_rate: float,
                     batch_size: int,
                     num_epochs: int,
                     patience: int,
                     learning_rate: float,
                     weight_decay: float,
                     device: torch.device,
                     include_size_formation: bool) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Train the model using K-fold cross-validation.
    
    Args:
        model: The model class (not instantiated)
        static_features: Static features array
        shape_error: Shape error array
        presence_error: Presence error array
        leader_off: Leader offset array
        targets: Targets array
        k: Number of folds
        time_steps: Number of time steps
        num_outputs: Number of outputs for temporal summary
        mode: Mode for temporal summary
        min_ratio: Minimum ratio for temporal summary
        hidden_sizes: List of hidden layer sizes
        output_size: Number of output targets
        dropout_rate: Dropout rate
        batch_size: Batch size
        num_epochs: Number of epochs
        patience: Early stopping patience
        learning_rate: Learning rate
        weight_decay: Weight decay
        device: Device to use for training
        include_size_formation: Whether to include size_formation feature
        
    Returns:
        Tuple[List[float], List[float], List[float], List[float]]: 
            List of training losses, validation losses, test losses, and R2 scores for each fold
    """
    # Initialize lists to store metrics
    train_losses_list = []
    val_losses_list = []
    test_losses_list = []
    r2_scores_list = []
    
    # Create K-fold cross-validation splitter
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # First split: separate test set (10%)
    train_val_idx, test_idx = train_test_split(
        np.arange(len(static_features)),
        test_size=0.1,
        random_state=42
    )
    
    # Get test data
    X_test_static = static_features[test_idx]
    X_test_shape = shape_error[test_idx]
    X_test_presence = presence_error[test_idx]
    X_test_leader = leader_off[test_idx]
    y_test = targets[test_idx]
    
    # Create test dataset and loader
    test_dataset = FormationDataset(X_test_static, X_test_shape, X_test_presence, X_test_leader, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Perform K-fold cross-validation on train_val data
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_idx), 1):
        print(f"\nTraining Fold {fold}/{k}")
        
        # Get train and validation data for this fold
        X_train_static = static_features[train_val_idx[train_idx]]
        X_train_shape = shape_error[train_val_idx[train_idx]]
        X_train_presence = presence_error[train_val_idx[train_idx]]
        X_train_leader = leader_off[train_val_idx[train_idx]]
        y_train = targets[train_val_idx[train_idx]]
        
        X_val_static = static_features[train_val_idx[val_idx]]
        X_val_shape = shape_error[train_val_idx[val_idx]]
        X_val_presence = presence_error[train_val_idx[val_idx]]
        X_val_leader = leader_off[train_val_idx[val_idx]]
        y_val = targets[train_val_idx[val_idx]]
        
        # Create datasets and dataloaders
        train_dataset = FormationDataset(X_train_static, X_train_shape, X_train_presence, X_train_leader, y_train)
        val_dataset = FormationDataset(X_val_static, X_val_shape, X_val_presence, X_val_leader, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Create input layer and model
        input_layer = InputLayer(time_steps, num_outputs, mode, min_ratio)
        model_instance = model(input_layer, hidden_sizes, output_size, dropout_rate, include_size_formation).to(device)
        
        # Define loss function and optimizer
        criterion = nn.SmoothL1Loss()
        optimizer = optim.AdamW(model_instance.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        # Train model
        train_losses, val_losses, _ = train_model(
            model_instance, train_loader, val_loader, criterion, optimizer, scheduler,
            num_epochs, patience, device
        )
        
        # Evaluate on test set
        test_loss, target_metrics = evaluate_model(model_instance, test_loader, criterion, device)
        
        # Store metrics
        train_losses_list.append(train_losses)
        val_losses_list.append(val_losses)
        test_losses_list.append(test_loss)
        r2_scores_list.append([metrics['r2'] for metrics in target_metrics.values()])
        
        # Save model for this fold
        os.makedirs('models', exist_ok=True)
        torch.save(model_instance.state_dict(), f"models/model_fold_{fold}.pth")
    
    return train_losses_list, val_losses_list, test_losses_list, r2_scores_list

def load_data(file_path: str, split_type: str = "a_priori", include_size_formation: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess the dataset from NPZ format.
    
    Args:
        file_path (str): Path to the dataset NPZ file
        split_type (str): Type of split to use ('a_priori', 'traditional', or 'k_fold')
        include_size_formation (bool): Whether to include the size_formation feature
        
    Returns:
        Tuple containing:
        - Training static features
        - Training shape error
        - Training presence error
        - Training leader offset
        - Training targets
        - Validation static features
        - Validation shape error
        - Validation presence error
        - Validation leader offset
        - Validation targets
        - Test static features
        - Test shape error
        - Test presence error
        - Test leader offset
        - Test targets
    """
    # Load the dataset
    data = np.load(file_path)
    
    # Filter out data beyond timestamp 300 (5 minutes)
    timestamps = data['timestamps']
    mask = timestamps <= 300.05
    data = {key: value[mask] if value.shape[0] == len(timestamps) else value for key, value in data.items()}
    
    time_steps = data['timestamps'].shape[0]
    
    # Extract static features based on user choice
    if include_size_formation:
        static_features = np.column_stack((data['size_formation'], data['comm_delay']))
    else:
        static_features = data['comm_delay'].reshape(-1, 1)  # Reshape to maintain 2D array
    
    # Extract time series data
    shape_error = data['shape_error']
    presence_error = data['presence_error']
    leader_off = data['leader_off']
    
    # Extract targets (comm_range, comm_failure)
    targets = np.column_stack((data['comm_range'], data['comm_failure']))
    
    if split_type == "a_priori":
        # Extract dataset types
        dataset_types = data['dataset_type']
        
        # Separate data by dataset type
        train_mask = dataset_types == TRAIN
        val_mask = dataset_types == VAL
        test_mask = dataset_types == TEST
        
        # Training data
        X_train_static = static_features[train_mask]
        X_train_shape = shape_error[train_mask]
        X_train_presence = presence_error[train_mask]
        X_train_leader = leader_off[train_mask]
        y_train = targets[train_mask]
        
        # Validation data
        X_val_static = static_features[val_mask]
        X_val_shape = shape_error[val_mask]
        X_val_presence = presence_error[val_mask]
        X_val_leader = leader_off[val_mask]
        y_val = targets[val_mask]
        
        # Test data
        X_test_static = static_features[test_mask]
        X_test_shape = shape_error[test_mask]
        X_test_presence = presence_error[test_mask]
        X_test_leader = leader_off[test_mask]
        y_test = targets[test_mask]
    else:
        # Use traditional split (for both traditional and k_fold)
        return split_dataset_traditional(static_features, shape_error, presence_error, leader_off, targets)
    
    return (X_train_static, X_train_shape, X_train_presence, X_train_leader, y_train,
            X_val_static, X_val_shape, X_val_presence, X_val_leader, y_val,
            X_test_static, X_test_shape, X_test_presence, X_test_leader, y_test)

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
        patience (int): Number of epochs to wait before early stopping
        device (torch.device): Device to use for training
        
    Returns:
        Tuple[List[float], List[float], int]: Training losses, validation losses, and best epoch
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for (static_features, shape_error, presence_error, leader_off), targets in train_loader:
            # Move data to device
            static_features = static_features.to(device)
            shape_error = shape_error.to(device)
            presence_error = presence_error.to(device)
            leader_off = leader_off.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(static_features, shape_error, presence_error, leader_off)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for (static_features, shape_error, presence_error, leader_off), targets in val_loader:
                # Move data to device
                static_features = static_features.to(device)
                shape_error = shape_error.to(device)
                presence_error = presence_error.to(device)
                leader_off = leader_off.to(device)
                targets = targets.to(device)
                
                outputs = model(static_features, shape_error, presence_error, leader_off)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            if old_lr != new_lr:
                print(f"Epoch [{epoch+1:3d}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {new_lr:.6f}")
            else:
                print(f"Epoch [{epoch+1:3d}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {old_lr:.6f}")
    
    return train_losses, val_losses, best_epoch

def evaluate_model(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate the model on the test set.
    
    Args:
        model (nn.Module): The trained model
        test_loader (DataLoader): DataLoader for test data
        criterion (nn.Module): Loss function
        device (torch.device): Device to use for evaluation
        
    Returns:
        Tuple[float, Dict[str, float]]: Overall test loss and dictionary of metrics for each target
    """
    model.eval()
    test_loss = 0.0
    all_predictions = []
    all_targets = []
    
    # Use tqdm for progress bar
    test_pbar = tqdm(test_loader, desc="Evaluating")
    with torch.no_grad():
        for (static_features, shape_error, presence_error, leader_off), targets in test_pbar:
            # Move data to device
            static_features = static_features.to(device)
            shape_error = shape_error.to(device)
            presence_error = presence_error.to(device)
            leader_off = leader_off.to(device)
            targets = targets.to(device)
            
            outputs = model(static_features, shape_error, presence_error, leader_off)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            test_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    test_loss /= len(test_loader)
    
    # Concatenate predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics for each target
    target_names = ['comm_range', 'comm_failure']
    target_metrics = {}
    for i, target_name in enumerate(target_names):
        mse = mean_squared_error(all_targets[:, i], all_predictions[:, i])
        r2 = r2_score(all_targets[:, i], all_predictions[:, i])
        target_metrics[target_name] = {
            'mse': mse,
            'r2': r2
        }
    
    return test_loss, target_metrics

def plot_losses(train_losses: List[float], val_losses: List[float], save_path: Optional[str] = None):
    """
    Plot training and validation losses.
    
    Args:
        train_losses (List[float]): List of training losses
        val_losses (List[float]): List of validation losses
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
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss plot saved to {save_path}")
    
    plt.show()
    plt.close()


def main():
    """
    Main function to execute the model training pipeline.
    """
    # Configuration
    DATASET_PATH = "dataset.npz"
    HIDDEN_SIZES = [128, 64, 32]
    DROPOUT_RATE = 0.1
    BATCH_SIZE = 64
    NUM_EPOCHS = 1000
    PATIENCE = 100
    LEARNING_RATE = 0.01
    WEIGHT_DECAY = 0.0001
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get temporal summary parameters from user
    num_outputs, mode, min_ratio = get_temporal_summary_params()
    
    # Get split type and K value from user
    split_type, k = get_split_type()
    
    # Ask about including size_formation feature
    include_size_formation = get_feature_choice()
    
    # Load the dataset
    print("\nLoading dataset...")
    data = np.load(DATASET_PATH)
    
    # Filter out data beyond timestamp 300 (5 minutes)
    timestamps = data['timestamps']
    mask = timestamps <= 300.05
    data = {key: value[mask] if value.shape[0] == len(timestamps) else value for key, value in data.items()}
    
    time_steps = data['timestamps'].shape[0]
    print(f"Time steps: {time_steps}")
    
    if split_type == "k_fold":
        # Load all data for k-fold cross-validation
        if include_size_formation:
            static_features = np.column_stack((data['size_formation'], data['comm_delay']))
        else:
            static_features = data['comm_delay'].reshape(-1, 1)  # Reshape to maintain 2D array
            
        shape_error = data['shape_error']
        presence_error = data['presence_error']
        leader_off = data['leader_off']
        targets = np.column_stack((data['comm_range'], data['comm_failure']))
        
        # Train with k-fold cross-validation
        train_losses_list, val_losses_list, test_losses_list, r2_scores_list = train_with_k_fold(
            FormationModel,
            static_features,
            shape_error,
            presence_error,
            leader_off,
            targets,
            k,
            time_steps,
            num_outputs,
            mode,
            min_ratio,
            HIDDEN_SIZES,
            2,  # output_size (comm_range and comm_failure)
            DROPOUT_RATE,
            BATCH_SIZE,
            NUM_EPOCHS,
            PATIENCE,
            LEARNING_RATE,
            WEIGHT_DECAY,
            device,
            include_size_formation
        )
        
        # Print results for each fold
        print("\nResults for each fold:")
        for fold in range(k):
            print(f"\nFold {fold + 1}:")
            print(f"Final Training Loss: {train_losses_list[fold][-1]:.4f}")
            print(f"Final Validation Loss: {val_losses_list[fold][-1]:.4f}")
            print(f"Test Loss: {test_losses_list[fold]:.4f}")
            print("R2 Scores:")
            for target, r2 in zip(['comm_range', 'comm_failure'], r2_scores_list[fold]):
                print(f"  {target}: {r2:.4f}")
        
        # Calculate mean and standard deviation for metrics across folds
        test_losses_array = np.array(test_losses_list)
        r2_scores_array = np.array(r2_scores_list)
        
        # Calculate mean R2 score for each fold (average across targets)
        mean_r2_per_fold = np.mean(r2_scores_array, axis=1)
        
        print("\nSummary Statistics Across Folds:")
        print(f"Test Loss - Mean: {np.mean(test_losses_array):.4f} ± {np.std(test_losses_array):.4f}")
        print(f"R2 Score (Average across targets) - Mean: {np.mean(mean_r2_per_fold):.4f} ± {np.std(mean_r2_per_fold):.4f}")
        
        # Plot losses for each fold
        plt.figure(figsize=(12, 6))
        for fold in range(k):
            plt.plot(train_losses_list[fold], label=f'Train Fold {fold + 1}')
            plt.plot(val_losses_list[fold], label=f'Val Fold {fold + 1}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses for Each Fold')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/loss_plot_kfold.png')
        plt.close()
        
    else:
        # Load and preprocess data with traditional or a priori split
        (X_train_static, X_train_shape, X_train_presence, X_train_leader, y_train,
         X_val_static, X_val_shape, X_val_presence, X_val_leader, y_val,
         X_test_static, X_test_shape, X_test_presence, X_test_leader, y_test) = load_data(DATASET_PATH, split_type, include_size_formation)
        
        # Print dataset sizes
        total_samples = len(X_train_static) + len(X_val_static) + len(X_test_static)
        train_pct = len(X_train_static) / total_samples * 100
        val_pct = len(X_val_static) / total_samples * 100
        test_pct = len(X_test_static) / total_samples * 100
        
        print("\nDataset sizes")
        print(f"Training:    {len(X_train_static):4d} samples ({train_pct:4.1f}%)")
        print(f"Validation:  {len(X_val_static):4d} samples ({val_pct:4.1f}%)")
        print(f"Test:        {len(X_test_static):4d} samples ({test_pct:4.1f}%)")
        
        # Create datasets and dataloaders
        train_dataset = FormationDataset(X_train_static, X_train_shape, X_train_presence, X_train_leader, y_train)
        val_dataset = FormationDataset(X_val_static, X_val_shape, X_val_presence, X_val_leader, y_val)
        test_dataset = FormationDataset(X_test_static, X_test_shape, X_test_presence, X_test_leader, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        # Create input layer
        input_layer = InputLayer(time_steps, num_outputs, mode, min_ratio)
        
        # Print temporal window sizes
        print("\nTemporal Window Sizes")
        print(f"Mode: {mode}, Number of windows: {num_outputs}, Minimum Ratio: {min_ratio}")
        print(f"Total Time: {time_steps*0.05:.2f} seconds")
        print("\nTemporal Window Sizes:")
        for i, window_size in enumerate(input_layer.shape_summary.window_sizes):
            print(f"  Window {i+1}: {window_size*0.05:.2f} seconds ({window_size/time_steps*100:.2f}% of total)")
        
        # Create model
        output_size = 2  # comm_range and comm_failure
        model = FormationModel(input_layer, HIDDEN_SIZES, output_size, DROPOUT_RATE, include_size_formation).to(device)
        
        # Define loss function and optimizer
        criterion = nn.SmoothL1Loss()  # Using SmoothL1Loss instead of MSELoss
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
        
        # Train model
        print("\nStarting model training...")
        train_losses, val_losses, best_epoch = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            NUM_EPOCHS, PATIENCE, device
        )
        
        # Plot losses
        plot_losses(train_losses, val_losses, "plots/loss_plot.png")
        
        # Evaluate model
        print("\nEvaluating model...")
        test_loss, target_metrics = evaluate_model(model, test_loader, criterion, device)
        
        # Print results
        print(f"\nTest Loss: {test_loss:.4f}")
        print("\nTarget Metrics:")
        for target, metrics in target_metrics.items():
            print(f"\n{target}:")
            print(f"MSE: {metrics['mse']:.4f}")
            print(f"R2 Score: {metrics['r2']:.4f}")
        
        # Save model
        torch.save(model.state_dict(), "models/model.pth")
        print("\nModel saved to models/model.pth")

if __name__ == "__main__":
    main() 