"""
Calculate communication metrics based on formation points, shape error, and communication delay.

This module provides functions to calculate communication-related metrics
(transmission range and failure rate) based on the number of formation points,
shape error values, and communication delay using deep learning.

Author: Laércio Lucchesi
Date: 2025
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from itertools import product

class CommunicationDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CommunicationNet(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=4, dropout_rate=0.3):
        super(CommunicationNet, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(hidden_size))
        self.layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers with decreasing size
        for i in range(num_layers - 1):
            current_size = hidden_size // (2 ** i) if i > 0 else hidden_size
            next_size = hidden_size // (2 ** (i + 1))
            self.layers.append(nn.Linear(current_size, next_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(next_size))
            self.layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_size // (2 ** (num_layers - 1)), 2))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, patience=20):
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            # Save the best model
            torch.save(best_model_state, 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    return best_val_loss, best_model_state

def grid_search(X, y, param_grid, device, n_splits=5):
    """
    Perform grid search to find the best hyperparameters.
    
    Args:
        X: Input features
        y: Target values
        param_grid: Dictionary of hyperparameters to search
        device: Device to run the model on
        n_splits: Number of folds for cross-validation
        
    Returns:
        dict: Best hyperparameters and their performance
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_params = None
    best_val_loss = float('inf')
    
    # Generate all combinations of parameters
    param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    
    print(f"\nStarting grid search with {len(param_combinations)} combinations...")
    
    for params in param_combinations:
        print(f"\nTesting parameters: {params}")
        fold_losses = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create datasets and dataloaders
            train_dataset = CommunicationDataset(X_train, y_train)
            val_dataset = CommunicationDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])
            
            # Initialize model
            model = CommunicationNet(
                input_size=3,
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout_rate=params['dropout_rate']
            ).to(device)
            
            # Initialize optimizer and scheduler
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=False
            )
            
            # Train model
            val_loss, _ = train_model(
                model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs=params['num_epochs'], device=device, patience=params['patience']
            )
            
            fold_losses.append(val_loss)
        
        # Calculate average validation loss across folds
        avg_val_loss = np.mean(fold_losses)
        print(f"Average validation loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_params = params
            print(f"New best parameters found! Loss: {best_val_loss:.4f}")
    
    return best_params, best_val_loss

def train_communication_metrics_model(metrics_data):
    """
    Train a deep learning model to predict communication metrics based on formation points,
    shape error, and communication delay.
    
    Args:
        metrics_data (pd.DataFrame): DataFrame containing the metrics data
        
    Returns:
        dict: Dictionary containing trained model and scalers
    """
    # Prepare input features and output targets
    X = metrics_data[[
        'formation_points_num_values',
        'shape_error',
        'communication_delay_values'
    ]].values
    
    y = metrics_data[[
        'communication_transmission_range_values',
        'communication_failure_rate_values'
    ]].values
    
    # Scale the data
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)
    
    # Define parameter grid for grid search
    param_grid = {
        'hidden_size': [64, 128, 256],
        'num_layers': [3, 4, 5],
        'dropout_rate': [0.2, 0.3, 0.4],
        'learning_rate': [0.0001, 0.001, 0.01],
        'weight_decay': [0.001, 0.01, 0.1],
        'batch_size': [32, 64, 128],
        'num_epochs': [100, 200],
        'patience': [10, 20]
    }
    
    # Perform grid search
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    best_params, best_val_loss = grid_search(X_scaled, y_scaled, param_grid, device)
    
    print("\nBest parameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Train final model with best parameters
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    train_dataset = CommunicationDataset(X_train, y_train)
    val_dataset = CommunicationDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'])
    
    model = CommunicationNet(
        input_size=3,
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout_rate=best_params['dropout_rate']
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Train final model
    best_val_loss, best_model_state = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=best_params['num_epochs'], device=device, patience=best_params['patience']
    )
    
    # Load best model state
    model.load_state_dict(best_model_state)
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_pred_scaled = model(X_val_tensor).cpu().numpy()
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        y_val_original = y_scaler.inverse_transform(y_val)
        
        mse = mean_squared_error(y_val_original, y_pred)
        r2 = r2_score(y_val_original, y_pred)
    
    return {
        'model': model,
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,
        'mse': mse,
        'r2': r2,
        'best_params': best_params
    }

def calculate_communication_metrics(formation_points, shape_error, communication_delay, model_data):
    """
    Calculate communication metrics based on formation points, shape error, and communication delay.
    
    Args:
        formation_points (float): Number of formation points
        shape_error (float): Shape error value
        communication_delay (float): Communication delay value
        model_data (dict): Dictionary containing trained model and scalers
        
    Returns:
        dict: Dictionary containing predicted communication metrics
    """
    # Prepare input features
    X = np.array([[formation_points, shape_error, communication_delay]])
    X_scaled = model_data['X_scaler'].transform(X)
    
    # Make prediction
    model = model_data['model']
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        y_pred_scaled = model(X_tensor).numpy()
        y_pred = model_data['y_scaler'].inverse_transform(y_pred_scaled)[0]
    
    return {
        'communication_transmission_range_values': y_pred[0],
        'communication_failure_rate_values': y_pred[1]
    }

def main():
    """
    Main function to demonstrate the usage of the communication metrics calculation.
    """
    # Read the metrics data
    metrics = pd.read_csv('metrics.csv')
    
    # Train the model
    model_data = train_communication_metrics_model(metrics)
    
    # Print model performance metrics
    print("\nModel Performance Metrics:")
    print(f"MSE: {model_data['mse']:.4f}")
    print(f"R²: {model_data['r2']:.4f}")
    
    # Example usage
    formation_points = 16
    shape_error = 0.3689575500831911
    communication_delay = 0.16  # Example delay value
    
    metrics = calculate_communication_metrics(formation_points, shape_error, communication_delay, model_data)
    
    print("\nPredicted Communication Metrics:")
    print(f"Formation Points: {formation_points}")
    print(f"Shape Error: {shape_error}")
    print(f"Communication Delay: {communication_delay}")
    print(f"Transmission Range: {metrics['communication_transmission_range_values']:.2f}")
    print(f"Failure Rate: {metrics['communication_failure_rate_values']:.2f}")

if __name__ == "__main__":
    main() 