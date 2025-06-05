"""
Machine Learning Module for Distributed Formation Control

This module implements Random Forest Regression to predict communication parameters
based on formation characteristics. It includes feature importance analysis and
model evaluation.

Key Features:
- Data loading and preprocessing
- Random Forest model training
- Feature importance analysis using both built-in and permutation methods
- Model evaluation using R² score
- Visualization of feature importances
- Model saving and loading

Dependencies:
    - numpy: Numerical computations
    - pandas: Data manipulation
    - scikit-learn: Machine learning tools
    - matplotlib: Plotting
    - joblib: Model persistence

Author: Laércio Lucchesi
Date: 2025-03-30
Version: 1.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
import os
import joblib
from typing import List, Tuple, Dict

def load_and_prepare_data(file_path: str, shape_error_type: str = 'avg_5min') -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Load and prepare data for machine learning.
    
    Args:
        file_path (str): Path to the dataset CSV file
        shape_error_type (str): Type of shape error to use
        
    Returns:
        Tuple containing:
        - DataFrame with prepared data
        - List of feature names
        - List of target names
    """
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Select features and targets
    features = ['size_formation', 'comm_delay', shape_error_type]
    targets = ['comm_range', 'comm_failure']
    
    # Create DataFrame with selected columns
    df = data[features + targets].copy()
    
    return df, features, targets

def train_random_forest(X: np.ndarray, y: np.ndarray, target_name: str) -> RandomForestRegressor:
    """
    Train a Random Forest model for a specific target.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        target_name (str): Name of the target variable
        
    Returns:
        RandomForestRegressor: Trained model
    """
    # Initialize model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    # Train model
    print(f"\nTraining Random Forest for {target_name}...")
    model.fit(X, y)
    
    return model

def analyze_feature_importance(model: RandomForestRegressor, X: np.ndarray, y: np.ndarray, 
                             feature_names: List[str], target_name: str) -> Dict:
    """
    Analyze feature importance using both built-in and permutation methods.
    
    Args:
        model: Trained Random Forest model
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        target_name: Name of the target variable
        
    Returns:
        Dictionary containing importance scores
    """
    # Built-in feature importance
    builtin_importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=True)
    
    # Permutation importance
    perm_importance = permutation_importance(
        model, X, y,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Calculate mean permutation importance
    perm_importance_mean = pd.Series(
        perm_importance.importances_mean,
        index=feature_names
    ).sort_values(ascending=True)
    
    return {
        'builtin': builtin_importance,
        'permutation': perm_importance_mean,
        'permutation_std': perm_importance.importances_std
    }

def plot_feature_importance(importance_dict: Dict, target_name: str, save_path: str = None):
    """
    Plot feature importance comparison between built-in and permutation methods.
    
    Args:
        importance_dict: Dictionary containing importance scores
        target_name: Name of the target variable
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Plot built-in importance
    plt.barh(
        importance_dict['builtin'].index,
        importance_dict['builtin'].values,
        alpha=0.8,
        label='Built-in'
    )
    
    # Plot permutation importance
    plt.barh(
        importance_dict['permutation'].index,
        importance_dict['permutation'].values,
        alpha=0.5,
        label='Permutation'
    )
    
    # Add error bars for permutation importance
    plt.errorbar(
        importance_dict['permutation'].values,
        importance_dict['permutation'].index,
        xerr=importance_dict['permutation_std'],
        fmt='none',
        color='black',
        capsize=5
    )
    
    plt.title(f'Feature Importance Comparison for {target_name}')
    plt.xlabel('Importance Score')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Feature importance plot saved to {save_path}")
    
    plt.show()

def main():
    """
    Main function to execute the machine learning pipeline.
    """
    # Configuration
    DATASET_PATH = 'dataset.csv'
    
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
    SHAPE_ERROR_TYPE = {
        "1": "avg_30s",
        "2": "avg_1min",
        "3": "avg_2min",
        "4": "avg_5min"
    }[choice]
    
    print(f"\nUsing {SHAPE_ERROR_TYPE} as shape error metric")
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Load and prepare data
    print(f"\nLoading data from {DATASET_PATH}...")
    df, features, targets = load_and_prepare_data(DATASET_PATH, SHAPE_ERROR_TYPE)
    
    # Split features and targets
    X = df[features].values
    y_dict = {target: df[target].values for target in targets}
    
    # Train and evaluate models for each target
    results = {}
    for target in targets:
        print(f"\nProcessing target: {target}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_dict[target],
            test_size=0.2,
            random_state=42
        )
        
        # Train model
        model = train_random_forest(X_train, y_train, target)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate R² score
        r2 = r2_score(y_test, y_pred)
        print(f"R² Score for {target}: {r2:.4f}")
        
        # Analyze feature importance
        importance_dict = analyze_feature_importance(
            model, X_test, y_test,
            features, target
        )
        
        # Plot feature importance
        plot_feature_importance(
            importance_dict,
            target,
            f'plots/feature_importance_{target}_{SHAPE_ERROR_TYPE}.png'
        )
        
        # Save model
        model_path = f'models/random_forest_{target}_{SHAPE_ERROR_TYPE}.joblib'
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        
        # Store results
        results[target] = {
            'model': model,
            'r2_score': r2,
            'importance': importance_dict
        }
    
    print("\nMachine learning pipeline completed successfully!")

if __name__ == "__main__":
    main() 