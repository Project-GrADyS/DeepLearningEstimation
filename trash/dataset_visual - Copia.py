"""
Dataset Visualization Module for Distributed Formation Control

This module is responsible for visualizing the relationships between features and targets
in the dataset using seaborn's pairplot, correlation heatmap, scatter plots, and 3D plots.

Key Features:
- Data loading
- Feature and target selection
- Visualization of relationships using pairplot
- Correlation matrix heatmap
- Scatter plots for each feature-target pair
- 3D plots showing how two features influence an output
- Saving plots to file

Dependencies:
    - numpy: Numerical computations
    - pandas: Data manipulation
    - seaborn: Statistical data visualization
    - matplotlib: Plotting

Author: Laércio Lucchesi
Date: 2025-03-30
Version: 1.0
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from typing import List, Tuple
from scipy.interpolate import griddata

def load_data(file_path: str, shape_error_type: str = 'avg_5min') -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Load the dataset and select features and targets.
    
    Args:
        file_path (str): Path to the dataset CSV file
        shape_error_type (str): Type of shape error to use ('avg_30s', 'avg_1min', 'avg_2min', 'avg_5min')
        
    Returns:
        Tuple[pd.DataFrame, List[str], List[str]]: 
            DataFrame with selected columns, list of feature names, list of target names
    """
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Select features and targets
    features = ['size_formation', 'comm_delay', shape_error_type]
    targets = ['comm_range', 'comm_failure']
    
    # Create a new DataFrame with only the selected columns
    df = data[features + targets].copy()
    
    return df, features, targets

def visualize_relationships(df: pd.DataFrame, features: List[str], targets: List[str], 
                           shape_error_type: str, save_path: str = None):
    """
    Visualize relationships between features and targets using seaborn's pairplot.
    
    Args:
        df (pd.DataFrame): DataFrame with features and targets
        features (List[str]): List of feature names
        targets (List[str]): List of target names
        shape_error_type (str): Type of shape error used
        save_path (str): Path to save the plot
    """
    # Set the style
    sns.set_style("whitegrid")
    
    # Create the pairplot
    plt.figure(figsize=(12, 10))
    sns.pairplot(df[features + targets], diag_kind='kde', plot_kws={'alpha': 0.6})
    
    # Add a title
    plt.suptitle(f"Relationships between Features and Targets\nShape Error: {shape_error_type}", 
                 y=1.02, fontsize=16)
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Pairplot saved to {save_path}")
    
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame, shape_error_type: str, save_path: str = None):
    """
    Create a correlation matrix heatmap.
    
    Args:
        df (pd.DataFrame): DataFrame with features and targets
        shape_error_type (str): Type of shape error used
        save_path (str): Path to save the plot
    """
    # Set the style
    sns.set_style("whitegrid")
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    
    # Add title
    plt.title(f"Correlation Matrix\nShape Error: {shape_error_type}", fontsize=16, pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Heatmap saved to {save_path}")
    
    plt.show()

def plot_feature_target_scatter(df: pd.DataFrame, features: List[str], targets: List[str], 
                               shape_error_type: str, save_path: str = None):
    """
    Create scatter plots for each feature against each target.
    
    Args:
        df (pd.DataFrame): DataFrame with features and targets
        features (List[str]): List of feature names
        targets (List[str]): List of target names
        shape_error_type (str): Type of shape error used
        save_path (str): Path to save the plot
    """
    # Set the style
    sns.set_style("whitegrid")
    
    # Calculate the number of rows and columns for the subplot grid
    n_features = len(features)
    n_targets = len(targets)
    
    # Create figure
    fig, axes = plt.subplots(n_features, n_targets, figsize=(15, 5*n_features))
    fig.suptitle(f"Feature vs Target Scatter Plots\nShape Error: {shape_error_type}", 
                 fontsize=16, y=1.02)
    
    # If there's only one feature or one target, axes will be 1D
    if n_features == 1 and n_targets == 1:
        axes = np.array([[axes]])
    elif n_features == 1:
        axes = np.array([axes])
    elif n_targets == 1:
        axes = np.array([[ax] for ax in axes])
    
    # Create scatter plots
    for i, feature in enumerate(features):
        for j, target in enumerate(targets):
            ax = axes[i, j]
            
            # Create scatter plot
            sns.scatterplot(data=df, x=feature, y=target, alpha=0.6, ax=ax)
            
            # Add regression line
            sns.regplot(data=df, x=feature, y=target, scatter=False, color='red', ax=ax)
            
            # Add labels
            ax.set_xlabel(feature)
            ax.set_ylabel(target)
            
            # Add correlation coefficient
            corr = df[feature].corr(df[target])
            ax.set_title(f"Correlation: {corr:.4f}")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Scatter plots saved to {save_path}")
    
    plt.show()

def plot_3d_feature_interaction(df: pd.DataFrame, features: List[str], targets: List[str], 
                               shape_error_type: str, save_path: str = None):
    """
    Create 3D plots showing how two features together influence an output.
    
    Args:
        df (pd.DataFrame): DataFrame with features and targets
        features (List[str]): List of feature names
        targets (List[str]): List of target names
        shape_error_type (str): Type of shape error used
        save_path (str): Path to save the plot
    """
    # Set the style
    plt.style.use('default')  # Reset style for 3D plots
    
    # Calculate the number of combinations
    n_features = len(features)
    n_targets = len(targets)
    n_combinations = n_features * (n_features - 1) // 2  # Number of unique feature pairs
    
    # Create figure with subplots in a 2x3 grid
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"3D Feature Interaction Plots\nShape Error: {shape_error_type}", 
                 fontsize=16, y=0.95)
    
    # Counter for subplot position
    plot_idx = 1
    
    # Create 3D plots for each feature pair and target
    for i in range(n_features):
        for j in range(i+1, n_features):
            feature1 = features[i]
            feature2 = features[j]
            
            for target in targets:
                # Create subplot
                ax = fig.add_subplot(2, 3, plot_idx, projection='3d')
                
                # Create scatter plot with smaller points and more transparency
                scatter = ax.scatter(df[feature1], df[feature2], df[target], 
                                   c=df[target], cmap='viridis', alpha=0.4, s=30)
                
                # Create a grid for the trend surface
                x_unique = np.linspace(df[feature1].min(), df[feature1].max(), 20)
                y_unique = np.linspace(df[feature2].min(), df[feature2].max(), 20)
                x_grid, y_grid = np.meshgrid(x_unique, y_unique)
                
                # Fit a 2D polynomial
                z_grid = griddata((df[feature1], df[feature2]), df[target], 
                                (x_grid, y_grid), method='cubic')
                
                # Plot surface with transparency
                surf = ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis',
                                     alpha=0.3, linewidth=0)
                
                # Add labels with smaller font
                ax.set_xlabel(feature1, fontsize=10)
                ax.set_ylabel(feature2, fontsize=10)
                ax.set_zlabel(target, fontsize=10)
                
                # Add title
                ax.set_title(f"{feature1} and {feature2} vs {target}", fontsize=12)
                
                # Add colorbar
                plt.colorbar(surf, ax=ax, label=target, shrink=0.5)
                
                # Set viewing angle
                ax.view_init(elev=20, azim=45)
                
                # Increment subplot counter
                plot_idx += 1
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"3D plots saved to {save_path}")
    
    plt.show()

def main():
    """
    Main function to execute the visualization pipeline.
    """
    # Configuration
    DATASET_PATH = 'dataset.csv'
    
    # Ask user to choose shape error type
    print("Choose the shape error metric to visualize:")
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
        print("Using avg_30s as shape error metric")
    elif choice == "2":
        SHAPE_ERROR_TYPE = 'avg_1min'
        print("Using avg_1min as shape error metric")
    elif choice == "3":
        SHAPE_ERROR_TYPE = 'avg_2min'
        print("Using avg_2min as shape error metric")
    elif choice == "4":
        SHAPE_ERROR_TYPE = 'avg_5min'
        print("Using avg_5min as shape error metric")
    
    # Create output directory
    os.makedirs('plots', exist_ok=True)
    
    # Load data
    print(f"Loading data from {DATASET_PATH}...")
    df, features, targets = load_data(DATASET_PATH, SHAPE_ERROR_TYPE)
    
    # Print dataset information
    print("\nDataset Information:")
    print(f"Number of samples: {len(df)}")
    print(f"Features: {features}")
    print(f"Targets: {targets}")
    
    # Print basic statistics
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Print correlation matrix
    print("\nCorrelation Matrix:")
    print(df.corr().round(4))
    
    # Visualize relationships with pairplot
    print("\nGenerating pairplot...")
    visualize_relationships(
        df, features, targets, SHAPE_ERROR_TYPE, 
        f'plots/pairplot_{SHAPE_ERROR_TYPE}.png'
    )
    
    # Create correlation heatmap
    print("\nGenerating correlation heatmap...")
    plot_correlation_heatmap(
        df, SHAPE_ERROR_TYPE, 
        f'plots/correlation_heatmap_{SHAPE_ERROR_TYPE}.png'
    )
    
    # Create feature-target scatter plots
    print("\nGenerating feature-target scatter plots...")
    plot_feature_target_scatter(
        df, features, targets, SHAPE_ERROR_TYPE, 
        f'plots/feature_target_scatter_{SHAPE_ERROR_TYPE}.png'
    )
    
    # Create 3D feature interaction plots
    print("\nGenerating 3D feature interaction plots...")
    plot_3d_feature_interaction(
        df, features, targets, SHAPE_ERROR_TYPE, 
        f'plots/3d_feature_interaction_{SHAPE_ERROR_TYPE}.png'
    )

if __name__ == "__main__":
    main() 