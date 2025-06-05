"""
Temporal Summary Layer Module for Deep Learning Models

This module provides a neural network layer that summarizes time series data using
different windowing strategies. It's particularly useful for processing sequential
data where you want to capture information at different time scales.

Key Features:
- Multiple windowing modes (short, linear, long)
- Configurable number of output windows
- Adjustable minimum ratio for window sizes
- Efficient tensor operations

Dependencies:
    - torch: Deep learning framework
    - numpy: Numerical computations

Author: Laércio Lucchesi
Date: 2025-03-30
Version: 1.0
"""

import torch
import torch.nn as nn
import numpy as np

class TemporalSummaryLayer(nn.Module):
    """
    Neural network layer that summarizes time series data using different windowing strategies.
    
    This layer takes a time series input and produces multiple summary statistics
    using different window sizes. The window sizes are determined by the chosen
    mode and minimum ratio parameters.
    """
    
    def __init__(self, time_steps: int, num_outputs: int, mode: str = 'linear', min_ratio: float = 0.1):
        """
        Initialize the temporal summary layer.
        
        Args:
            time_steps (int): Total time steps in the input series
            num_outputs (int): Number of summary windows to generate
            mode (str): Windowing mode, one of 'short', 'linear', or 'long'
            min_ratio (float): Minimum alpha as percentage of max (1.0)
        """
        super().__init__()
        self.time_steps = time_steps
        self.num_outputs = num_outputs
        self.mode = mode.lower()
        self.min_ratio = min_ratio

        x = np.arange(1, num_outputs + 1)

        # Compute base alpha values
        if self.mode == 'short':
            alpha_base = 1 / x
        elif self.mode == 'linear':
            alpha_base = 1 - (x - 1) / num_outputs
        elif self.mode == 'long':
            alpha_base = 1 + 1 / num_outputs - 1 / (num_outputs - x + 1)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        # Normalize alpha to range [min_ratio, 1]
        alpha_norm = (alpha_base - alpha_base.min()) / (alpha_base.max() - alpha_base.min())
        alpha_scaled = min_ratio + (1.0 - min_ratio) * alpha_norm

        self.window_sizes = [max(1, int(round(time_steps * a))) for a in alpha_scaled]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the input tensor through the temporal summary layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, time_steps)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_outputs)
        """
        outputs = []
        for window_size in self.window_sizes:
            slice_ = x[:, :, -window_size:]
            avg = slice_.mean(dim=2)
            outputs.append(avg)
        return torch.cat(outputs, dim=1)
