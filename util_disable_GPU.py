"""
GPU Disabling Utility for PyTorch Applications

This utility provides a simple way to disable GPU usage in PyTorch applications
and ensure all computations are performed on CPU. It's particularly useful when:
- Running on systems without GPU
- Need to ensure consistent CPU-only execution
- Want to avoid GPU memory issues
- Need to debug GPU-related problems

Purpose:
- Force CPU-only execution
- Disable CUDA device visibility
- Provide system resource information
- Ensure consistent behavior across different systems

Features:
- Sets PyTorch device to CPU
- Disables CUDA device visibility
- Reports GPU availability status
- Shows available CPU cores

Usage:
    python disable_GPU.py
    # The script will:
    # 1. Set device to CPU
    # 2. Disable GPU visibility
    # 3. Display system resource information

Dependencies:
    - torch: PyTorch library
    - os: Operating system interface
    - multiprocessing: CPU core information

Author: Laércio Lucchesi
Date: 2025-03-30
Version: 1.0
"""

import torch
import os
import multiprocessing

# Set device to CPU
device = torch.device("cpu")

# Disable GPU usage by setting CUDA_VISIBLE_DEVICES to -1
# This prevents PyTorch from seeing any GPU devices
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Print system resource information
print("CUDA available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
print("Device:", device)
print(f"Number of available CPU cores: {multiprocessing.cpu_count()}")
