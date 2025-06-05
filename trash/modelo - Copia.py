import torch
import pandas as pd
import numpy as np
import os
from typing import List
from sklearn.metrics import mean_squared_error, r2_score

# Define the FormationModel class
class FormationModel(torch.nn.Module):
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
            layers.append(torch.nn.Linear(prev_size, hidden_size))
            layers.append(torch.nn.LeakyReLU(0.1))  # Using LeakyReLU instead of ReLU
            if dropout_rate > 0:
                layers.append(torch.nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(torch.nn.Linear(prev_size, output_size))
        
        # Combine all layers
        self.network = torch.nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return self.network(x)

# Path to the model file
model_path = 'models/model_baseline_avg_5min_feature_size_formation_feature_comm_delay_feature_new_target_01_target_new_target_02_target.pth'

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit(1)

# Load the dataset and model
try:
    # Load the dataset
    dataset_path = 'dataset.csv'
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        exit(1)
    
    data = pd.read_csv(dataset_path)
    
    # Filter test data
    test_data = data[data['dataset_type'] == 'test']
    print(f"Number of test samples: {len(test_data)}")
    
    # Load the model checkpoint
    checkpoint = torch.load(model_path)
    
    # Print model structure
    print("\nModel Structure:")
    print(checkpoint['model_state_dict'])
    
    # Print model configuration
    print("\nModel Configuration:")
    config = checkpoint['config']
    print(f"Input size: {config['input_size']}")
    print(f"Hidden sizes: {config['hidden_sizes']}")
    print(f"Output size: {config['output_size']}")
    print(f"Dropout rate: {config['dropout_rate']}")
    
    # Print roles
    print("\nVariable Roles:")
    for var, role in config['roles'].items():
        print(f"{var}: {role}")
    
    # Extract model configuration
    input_size = config['input_size']
    hidden_sizes = config['hidden_sizes']
    output_size = config['output_size']
    dropout_rate = config['dropout_rate']
    
    # Create model with the same configuration
    model = FormationModel(input_size, hidden_sizes, output_size, dropout_rate)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    
    # Get feature columns based on roles
    feature_columns = [var for var, role in config['roles'].items() if role == 'feature']
    print(f"\nFeature columns: {feature_columns}")
    
    # Extract features from test data
    X_test = test_data[feature_columns].values
    
    # Convert to tensor
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(X_test_tensor)
    
    # Convert predictions to numpy array
    predictions = predictions.numpy()
    
    # Get target columns based on roles
    target_columns = [var for var, role in config['roles'].items() if role == 'target']
    print(f"Target columns: {target_columns}")
    
    # Create a DataFrame with the predictions
    results_df = pd.DataFrame(predictions, columns=target_columns)
    
    # Add original features to the results
    for col in feature_columns:
        results_df[col] = test_data[col].values
    
    # Add the actual values of comm_range and comm_failure
    results_df['comm_range'] = test_data['comm_range'].values
    results_df['comm_failure'] = test_data['comm_failure'].values
    
    # Calculate failure_pred and range_pred using the provided equations
    print("\nCalculating failure_pred and range_pred...")
    
    # Add a small constant to avoid division by zero
    epsilon = 1e-10
    
    # Calculate failure_pred
    results_df['failure_pred'] = (
        results_df['new_target_02'] - 
        (0.3154 + 0.4867 * results_df['size_formation'] + 0.0721 * results_df['comm_delay'])
    ) / (
        0.1037 - 0.0897 * results_df['size_formation'] * results_df['comm_delay'] * 
        ((1 - results_df['new_target_01']) / (results_df['new_target_01'] + epsilon))
    )
    
    # Calculate range_pred
    results_df['range_pred'] = (
        results_df['size_formation'] * 
        results_df['comm_delay'] * 
        results_df['failure_pred'] * 
        ((1 - results_df['new_target_01']) / (results_df['new_target_01'] + epsilon))
    )
    
    # Handle any NaN or infinite values
    results_df['failure_pred'] = results_df['failure_pred'].replace([np.inf, -np.inf], np.nan)
    results_df['range_pred'] = results_df['range_pred'].replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with 0
    results_df['failure_pred'] = results_df['failure_pred'].fillna(0)
    results_df['range_pred'] = results_df['range_pred'].fillna(0)
    
    # Calculate MSE and R² score for failure_pred vs comm_failure
    failure_mse = mean_squared_error(test_data['comm_failure'], results_df['failure_pred'])
    failure_r2 = r2_score(test_data['comm_failure'], results_df['failure_pred'])
    
    # Calculate MSE and R² score for range_pred vs comm_range
    range_mse = mean_squared_error(test_data['comm_range'], results_df['range_pred'])
    range_r2 = r2_score(test_data['comm_range'], results_df['range_pred'])
    
    # Print the metrics
    print("\nMetrics for derived variables:")
    print(f"Failure (comm_failure):")
    print(f"  MSE: {failure_mse:.6f}")
    print(f"  R² Score: {failure_r2:.6f}")
    print(f"Range (comm_range):")
    print(f"  MSE: {range_mse:.6f}")
    print(f"  R² Score: {range_r2:.6f}")
    
    # Save results to CSV
    results_df.to_csv('test_predictions.csv', index=False)
    print(f"\nPredictions saved to test_predictions.csv")
    
    # Print some statistics
    print("\nPrediction Statistics:")
    for col in target_columns + ['failure_pred', 'range_pred', 'comm_range', 'comm_failure']:
        print(f"{col}:")
        print(f"  Min: {results_df[col].min():.6f}")
        print(f"  Max: {results_df[col].max():.6f}")
        print(f"  Mean: {results_df[col].mean():.6f}")
        print(f"  Std: {results_df[col].std():.6f}")
    
    # Print first 5 predictions
    print("\nFirst 5 predictions:")
    print(results_df.head())
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 