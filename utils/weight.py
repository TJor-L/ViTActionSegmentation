import torch
import numpy as np


def smooth_class_weights(class_counts, smoothing_factor=0.5, device='cpu', verbose=False):
    """
    Smooth class weights to avoid extreme differences while preserving some differentiation.

    Parameters:
    - class_counts (numpy array or list): Array of class counts.
    - smoothing_factor (float): Factor for smoothing; higher values lead to more uniform weights (0 < smoothing_factor <= 1).
    - device (str): Device for the torch tensor (e.g., 'cpu' or 'cuda').

    Returns:
    - torch.Tensor: Smoothed weights as a tensor.
    """
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / \
        np.sum(class_weights) * len(class_counts)
    
    if verbose:    
        print("Weight before smooth", class_weights)

    class_weights = np.log(1 + smoothing_factor * class_weights)
    class_weights = class_weights / \
        np.sum(class_weights) * len(class_counts)
    if verbose:    
        print("Weight after smooth", class_weights)
    return torch.tensor(class_weights, dtype=torch.float, device=device)
