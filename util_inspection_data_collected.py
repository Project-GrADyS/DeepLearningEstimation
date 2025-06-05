"""
Data Inspection Utility for Simulation Results

This utility provides a simple way to inspect and visualize simulation data
from a single simulation run. It creates three separate plots:
1. One showing the shape error signal analysis
2. Another showing the leader_off signal analysis (if available)
3. A third showing the presence error signal analysis (if available)

Each plot includes:
- Original signal
- Cumulative mean
- Moving averages for different time windows (30s, 1min, 2min, 5min)

Purpose:
- Quick inspection of simulation results
- Visual verification of data quality
- Analysis of shape error, leader status, and presence error behavior over time
- Detection of potential issues in the simulation data

Features:
- Separate plots for shape_error, leader_off, and presence_error analysis
- Original signal visualization
- Cumulative mean tracking
- Moving averages for different time windows
- Saves processed data to CSV for further analysis

Usage:
    python inspection_data_collected.py
    # The script will:
    # 1. Read the specified CSV file
    # 2. Calculate moving averages
    # 3. Display three separate plots
    # 4. Show basic statistics

Dependencies:
    - pandas: Data manipulation
    - numpy: Numerical computations
    - matplotlib: Plotting

Author: Laércio Lucchesi
Date: 2025-03-30
Version: 1.2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants for data processing
SAMPLING_RATE = 0.05  # seconds
WINDOW_SIZES = {
    '30s': 30,
    '1min': 60,
    '2min': 120,
    '5min': 300
}

def calculate_moving_averages(df, sampling_rate, window_sizes):
    """
    Calculates moving averages for different time windows.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        sampling_rate (float): Sampling rate in seconds
        window_sizes (dict): Dictionary with window sizes in seconds
    """
    # Calculate moving averages for shape_error
    for name, seconds in window_sizes.items():
        window_size = int(seconds/sampling_rate)
        df[f'ma_{name}'] = df['shape_error'].rolling(
            window=window_size, 
            min_periods=window_size
        ).mean()
    
    # Calculate moving averages for leader_off if the column exists
    if 'leader_off' in df.columns:
        for name, seconds in window_sizes.items():
            window_size = int(seconds/sampling_rate)
            df[f'ma_leader_off_{name}'] = df['leader_off'].rolling(
                window=window_size, 
                min_periods=window_size
            ).mean()
    
    # Calculate moving averages for presence_error if the column exists
    if 'presence_error' in df.columns:
        for name, seconds in window_sizes.items():
            window_size = int(seconds/sampling_rate)
            df[f'ma_presence_error_{name}'] = df['presence_error'].rolling(
                window=window_size, 
                min_periods=window_size
            ).mean()

def plot_time_series(df, window_sizes):
    """
    Creates three separate plots:
    1. One with subplots showing the shape error signal
    2. Another with subplots showing the leader_off signal (if available)
    3. A third with subplots showing the presence_error signal (if available)
    
    Each plot includes:
    - Original signal
    - Cumulative mean
    - Moving averages for different time windows
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        window_sizes (dict): Dictionary with window sizes
    """
    # Check if leader_off and presence_error columns exist
    has_leader_off = 'leader_off' in df.columns
    has_presence_error = 'presence_error' in df.columns
    
    # Create first figure for shape_error
    fig1, axes1 = plt.subplots(4, 1, figsize=(15, 20))
    fig1.suptitle('Shape Error Analysis', fontsize=16, y=0.95)
    
    # Common settings for shape_error subplots
    for ax in axes1:
        ax.grid(True, alpha=0.3)
    
    # Remove x-axis labels from all plots except the last one
    for ax in axes1[:-1]:
        ax.set_xticklabels([])
    
    # Set x-axis label only for the last plot
    axes1[-1].set_xlabel('Time (s)')

    # Plot Shape Error graphs
    # Plot 1: Shape Error, Cumulative Mean and 30s Moving Average
    axes1[0].set_ylabel('Shape Error')
    axes1[0].plot(df['timestamp'], df['shape_error'], 
                 label='Shape Error', alpha=0.3, color='gray')
    axes1[0].plot(df['timestamp'], df['cumulative_mean'], 
                 label='Cumulative Mean', color='darkgray', linewidth=2)
    axes1[0].plot(df['timestamp'], df['ma_30s'], 
                 label='Moving Average (30s)', color='red', linewidth=2)
    axes1[0].legend()

    # Plot 2: Shape Error, Cumulative Mean and 1min Moving Average
    axes1[1].set_ylabel('Shape Error')
    axes1[1].plot(df['timestamp'], df['shape_error'], 
                label='Shape Error', alpha=0.3, color='gray')
    axes1[1].plot(df['timestamp'], df['cumulative_mean'], 
                 label='Cumulative Mean', color='darkgray', linewidth=2)
    axes1[1].plot(df['timestamp'], df['ma_1min'], 
                 label='Moving Average (1min)', color='green', linewidth=2)
    axes1[1].legend()
    
    # Plot 3: Shape Error, Cumulative Mean and 2min Moving Average
    axes1[2].set_ylabel('Shape Error')
    axes1[2].plot(df['timestamp'], df['shape_error'], 
                label='Shape Error', alpha=0.3, color='gray')
    axes1[2].plot(df['timestamp'], df['cumulative_mean'], 
                 label='Cumulative Mean', color='darkgray', linewidth=2)
    axes1[2].plot(df['timestamp'], df['ma_2min'], 
                 label='Moving Average (2min)', color='blue', linewidth=2)
    axes1[2].legend()
    
    # Plot 4: Shape Error, Cumulative Mean and 5min Moving Average
    axes1[3].set_ylabel('Shape Error')
    axes1[3].plot(df['timestamp'], df['shape_error'], 
                label='Shape Error', alpha=0.3, color='gray')
    axes1[3].plot(df['timestamp'], df['cumulative_mean'], 
                 label='Cumulative Mean', color='darkgray', linewidth=2)
    axes1[3].plot(df['timestamp'], df['ma_5min'], 
                 label='Moving Average (5min)', color='black', linewidth=2)
    axes1[3].legend()
    
    # Adjust layout for first figure
    plt.figure(fig1.number)
    plt.subplots_adjust(bottom=0.1, hspace=0.3)
    
    # Create second figure for leader_off if the column exists
    if has_leader_off:
        fig2, axes2 = plt.subplots(4, 1, figsize=(15, 20))
        fig2.suptitle('Leader Off Analysis', fontsize=16, y=0.95)
        
        # Common settings for leader_off subplots
        for ax in axes2:
            ax.grid(True, alpha=0.3)
        
        # Remove x-axis labels from all plots except the last one
        for ax in axes2[:-1]:
            ax.set_xticklabels([])
        
        # Set x-axis label only for the last plot
        axes2[-1].set_xlabel('Time (s)')

        # Plot Leader Off graphs
        # Plot 1: Leader Off, Cumulative Mean and 30s Moving Average
        axes2[0].set_ylabel('Leader Off')
        axes2[0].plot(df['timestamp'], df['leader_off'], 
                     label='Leader Off', alpha=0.3, color='gray')
        axes2[0].plot(df['timestamp'], df['leader_off'].expanding().mean(), 
                     label='Cumulative Mean', color='darkgray', linewidth=2)
        axes2[0].plot(df['timestamp'], df['ma_leader_off_30s'], 
                     label='Moving Average (30s)', color='red', linewidth=2)
        axes2[0].legend()

        # Plot 2: Leader Off, Cumulative Mean and 1min Moving Average
        axes2[1].set_ylabel('Leader Off')
        axes2[1].plot(df['timestamp'], df['leader_off'], 
                    label='Leader Off', alpha=0.3, color='gray')
        axes2[1].plot(df['timestamp'], df['leader_off'].expanding().mean(), 
                     label='Cumulative Mean', color='darkgray', linewidth=2)
        axes2[1].plot(df['timestamp'], df['ma_leader_off_1min'], 
                     label='Moving Average (1min)', color='green', linewidth=2)
        axes2[1].legend()
        
        # Plot 3: Leader Off, Cumulative Mean and 2min Moving Average
        axes2[2].set_ylabel('Leader Off')
        axes2[2].plot(df['timestamp'], df['leader_off'], 
                    label='Leader Off', alpha=0.3, color='gray')
        axes2[2].plot(df['timestamp'], df['leader_off'].expanding().mean(), 
                     label='Cumulative Mean', color='darkgray', linewidth=2)
        axes2[2].plot(df['timestamp'], df['ma_leader_off_2min'], 
                     label='Moving Average (2min)', color='blue', linewidth=2)
        axes2[2].legend()
        
        # Plot 4: Leader Off, Cumulative Mean and 5min Moving Average
        axes2[3].set_ylabel('Leader Off')
        axes2[3].plot(df['timestamp'], df['leader_off'], 
                    label='Leader Off', alpha=0.3, color='gray')
        axes2[3].plot(df['timestamp'], df['leader_off'].expanding().mean(), 
                     label='Cumulative Mean', color='darkgray', linewidth=2)
        axes2[3].plot(df['timestamp'], df['ma_leader_off_5min'], 
                     label='Moving Average (5min)', color='black', linewidth=2)
        axes2[3].legend()
        
        # Adjust layout for second figure
        plt.figure(fig2.number)
        plt.subplots_adjust(bottom=0.1, hspace=0.3)
    
    # Create third figure for presence_error if the column exists
    if has_presence_error:
        fig3, axes3 = plt.subplots(4, 1, figsize=(15, 20))
        fig3.suptitle('Presence Error Analysis', fontsize=16, y=0.95)
        
        # Common settings for presence_error subplots
        for ax in axes3:
            ax.grid(True, alpha=0.3)
        
        # Remove x-axis labels from all plots except the last one
        for ax in axes3[:-1]:
            ax.set_xticklabels([])
        
        # Set x-axis label only for the last plot
        axes3[-1].set_xlabel('Time (s)')

        # Plot Presence Error graphs
        # Plot 1: Presence Error, Cumulative Mean and 30s Moving Average
        axes3[0].set_ylabel('Presence Error')
        axes3[0].plot(df['timestamp'], df['presence_error'], 
                     label='Presence Error', alpha=0.3, color='gray')
        axes3[0].plot(df['timestamp'], df['presence_error'].expanding().mean(), 
                     label='Cumulative Mean', color='darkgray', linewidth=2)
        axes3[0].plot(df['timestamp'], df['ma_presence_error_30s'], 
                     label='Moving Average (30s)', color='red', linewidth=2)
        axes3[0].legend()

        # Plot 2: Presence Error, Cumulative Mean and 1min Moving Average
        axes3[1].set_ylabel('Presence Error')
        axes3[1].plot(df['timestamp'], df['presence_error'], 
                    label='Presence Error', alpha=0.3, color='gray')
        axes3[1].plot(df['timestamp'], df['presence_error'].expanding().mean(), 
                     label='Cumulative Mean', color='darkgray', linewidth=2)
        axes3[1].plot(df['timestamp'], df['ma_presence_error_1min'], 
                     label='Moving Average (1min)', color='green', linewidth=2)
        axes3[1].legend()
        
        # Plot 3: Presence Error, Cumulative Mean and 2min Moving Average
        axes3[2].set_ylabel('Presence Error')
        axes3[2].plot(df['timestamp'], df['presence_error'], 
                    label='Presence Error', alpha=0.3, color='gray')
        axes3[2].plot(df['timestamp'], df['presence_error'].expanding().mean(), 
                     label='Cumulative Mean', color='darkgray', linewidth=2)
        axes3[2].plot(df['timestamp'], df['ma_presence_error_2min'], 
                     label='Moving Average (2min)', color='blue', linewidth=2)
        axes3[2].legend()
        
        # Plot 4: Presence Error, Cumulative Mean and 5min Moving Average
        axes3[3].set_ylabel('Presence Error')
        axes3[3].plot(df['timestamp'], df['presence_error'], 
                    label='Presence Error', alpha=0.3, color='gray')
        axes3[3].plot(df['timestamp'], df['presence_error'].expanding().mean(), 
                     label='Cumulative Mean', color='darkgray', linewidth=2)
        axes3[3].plot(df['timestamp'], df['ma_presence_error_5min'], 
                     label='Moving Average (5min)', color='black', linewidth=2)
        axes3[3].legend()
        
        # Adjust layout for third figure
        plt.figure(fig3.number)
        plt.subplots_adjust(bottom=0.1, hspace=0.3)
    
    # Show all figures
    plt.show()

def main():
    """
    Main function to process and visualize simulation data.
    
    This function:
    1. Reads the CSV file
    2. Calculates cumulative mean
    3. Computes moving averages
    4. Saves processed data
    5. Creates three separate visualizations
    6. Displays basic statistics
    """
    # Read CSV file
    # Pay attention to the file path !!!
    # You should change this file path to the one you want to inspect
    #file_path = ".\dataset_validation\P124R059D170F021.csv" 
    file_path = "P124R059D170F021.csv" 
    df = pd.read_csv(file_path)
    
    # Print available columns
    print("\nAvailable columns in the dataset:")
    print(df.columns.tolist())
    
    # Calculate cumulative mean
    df['cumulative_mean'] = df['shape_error'].expanding().mean()
    
    # Calculate moving averages
    calculate_moving_averages(df, SAMPLING_RATE, WINDOW_SIZES)
    
    # Save processed dataframe
    df.to_csv('time_series.csv', index=False)
    
    # Plot results
    plot_time_series(df, WINDOW_SIZES)
    
    # Display basic statistics
    print("\nBasic statistics:")
    print(df.describe())

if __name__ == "__main__":
    main() 