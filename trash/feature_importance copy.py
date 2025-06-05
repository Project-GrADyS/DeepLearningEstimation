"""
Feature Importance Module using Captum and SHAP

Calculates and visualizes feature importance for a PyTorch model
using DeepLiftShap from the Captum library or SHAP library.

Author: Laércio Lucchesi
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from captum.attr import DeepLiftShap
import warnings
import shap

def compute_feature_importance_captum(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    features: list,
    baseline: str = "random",
    baseline_samples: int = 100,
    target_idx: int = 0,
    target_name: str = None
) -> np.ndarray:
    """
    Compute feature importance using DeepLiftShap from Captum.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        inputs (torch.Tensor): Input tensor (batch x features).
        features (list): List of feature names.
        baseline (str): 'zero', 'mean', or 'random'. Determines the baseline for attribution.
        baseline_samples (int): Number of baseline samples to use (if random).
        target_idx (int): Index of the target output to use for attribution (if model has multiple outputs).
        target_name (str): Name of the target variable for the title of the plot.

    Returns:
        np.ndarray: Mean attribution for each feature.
    """
    model.eval()
    
    # Get the device of the model
    device = next(model.parameters()).device
    
    # Move inputs to the same device as the model
    inputs = inputs.to(device)
    
    # Create baseline based on the selected option
    if baseline == "mean":
        base = torch.mean(inputs, dim=0, keepdim=True).repeat(baseline_samples, 1)
    elif baseline == "zero":
        base = torch.zeros((baseline_samples, inputs.shape[1]), dtype=inputs.dtype, device=device)
    else:  # 'random'
        indices = torch.randint(0, inputs.shape[0], (baseline_samples,), device=device)
        base = inputs[indices]
    
    # Ensure baseline is on the same device as the model
    base = base.to(device)

    # Create a wrapper for the model to handle multiple outputs
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model, target_idx):
            super().__init__()
            self.model = model
            self.target_idx = target_idx
            
        def forward(self, x):
            output = self.model(x)
            # If output is a tensor with multiple values, select the target index
            if isinstance(output, torch.Tensor) and output.dim() > 1 and output.shape[1] > 1:
                return output[:, self.target_idx]
            return output
    
    # Wrap the model to handle multiple outputs
    wrapped_model = ModelWrapper(model, target_idx)
    
    # Suppress the specific warning about forward/backward hooks
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Setting forward, backward hooks and attributes on non-linear")
        explainer = DeepLiftShap(wrapped_model)
        attributions = explainer.attribute(inputs, baselines=base)

    mean_attr = attributions.mean(dim=0).detach().cpu().numpy()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(features[::-1], mean_attr[::-1])
    plt.xlabel("Feature Importance (DeepLiftShap)")
    
    # Use target name in title if provided
    if target_name:
        plt.title(f"Captum Feature Importance for {target_name}")
    else:
        plt.title(f"Captum Feature Importance (Target {target_idx})")
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return mean_attr

def compute_feature_importance_shap(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    features: list,
    background_samples: int = 100,
    target_idx: int = 0,
    target_name: str = None
) -> np.ndarray:
    """
    Compute feature importance using SHAP library.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        inputs (torch.Tensor): Input tensor (batch x features).
        features (list): List of feature names.
        background_samples (int): Number of background samples to use.
        target_idx (int): Index of the target output to use for attribution (if model has multiple outputs).
        target_name (str): Name of the target variable for the title of the plot.

    Returns:
        np.ndarray: Mean attribution for each feature.
    """
    model.eval()
    
    # Get the device of the model
    device = next(model.parameters()).device
    
    # Move inputs to the same device as the model
    inputs = inputs.to(device)
    
    # Create a wrapper for the model to handle multiple outputs
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model, target_idx):
            super().__init__()
            self.model = model
            self.target_idx = target_idx
            
        def forward(self, x):
            output = self.model(x)
            # If output is a tensor with multiple values, select the target index
            if isinstance(output, torch.Tensor) and output.dim() > 1 and output.shape[1] > 1:
                return output[:, self.target_idx]
            return output
    
    # Wrap the model to handle multiple outputs
    wrapped_model = ModelWrapper(model, target_idx)
    
    # Convert inputs to numpy for SHAP
    inputs_np = inputs.detach().cpu().numpy()
    
    # Select background samples
    if background_samples < inputs_np.shape[0]:
        background = inputs_np[:background_samples]
    else:
        background = inputs_np
    
    # Create a function that wraps the model for SHAP
    def model_fn(x):
        # Convert numpy array to tensor
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        # Get predictions
        with torch.no_grad():
            output = wrapped_model(x_tensor)
        # Convert to numpy and return
        return output.detach().cpu().numpy()
    
    # Create SHAP explainer with explicit RNG
    rng = np.random.default_rng(42)  # Use the new recommended way to create an RNG
    explainer = shap.KernelExplainer(model_fn, background, link="identity", seed=rng)
    
    # Calculate SHAP values for a subset of the data
    # We limit to 100 samples to avoid memory issues
    max_samples = min(100, inputs_np.shape[0])
    shap_values = explainer.shap_values(inputs_np[:max_samples])
    
    # If shap_values is a list (for multi-output models), select the target index
    if isinstance(shap_values, list):
        shap_values = shap_values[target_idx]
    
    # Calculate mean absolute SHAP values for each feature
    mean_attr = np.abs(shap_values).mean(axis=0)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(features[::-1], mean_attr[::-1])
    plt.xlabel("Feature Importance (|SHAP|)")
    
    # Use target name in title if provided
    if target_name:
        plt.title(f"SHAP Feature Importance for {target_name}")
    else:
        plt.title(f"SHAP Feature Importance (Target {target_idx})")
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Also plot the SHAP summary plot
    plt.figure(figsize=(10, 6))
    # Suppress the warning about NumPy RNG
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        shap.summary_plot(shap_values, inputs_np[:max_samples], feature_names=features, show=False)
    plt.title(f"SHAP Summary Plot for {target_name if target_name else f'Target {target_idx}'}")
    plt.tight_layout()
    plt.show()

    return mean_attr

def compute_feature_importance(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    features: list,
    method: str = "captum",
    baseline: str = "random",
    baseline_samples: int = 50,
    target_idx: int = 0,
    target_name: str = None
) -> np.ndarray:
    """
    Compute feature importance using either Captum or SHAP.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        inputs (torch.Tensor): Input tensor (batch x features).
        features (list): List of feature names.
        method (str): 'captum' or 'shap'. Determines which library to use.
        baseline (str): 'zero', 'mean', or 'random'. Determines the baseline for attribution (Captum only).
        baseline_samples (int): Number of baseline samples to use (if random) (Captum only).
        target_idx (int): Index of the target output to use for attribution (if model has multiple outputs).
        target_name (str): Name of the target variable for the title of the plot.

    Returns:
        np.ndarray: Mean attribution for each feature.
    """
    if method.lower() == "captum":
        return compute_feature_importance_captum(
            model, inputs, features, baseline, baseline_samples, target_idx, target_name
        )
    elif method.lower() == "shap":
        return compute_feature_importance_shap(
            model, inputs, features, baseline_samples, target_idx, target_name
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'captum' or 'shap'.")
