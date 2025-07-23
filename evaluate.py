# Additional tests for Dataloader.py

import os
import math
import pytest
import numpy as np
import pandas as pd
import tempfile
from Dataloader_back import preprocess_log_file

def euclidean_dist(p1, p2):
    """
    Calculate Euclidean distance between two points in 3D space.
    
    Args:
        p1: First point coordinates (x, y, z)
        p2: Second point coordinates (x, y, z)
    
    Returns:
        Euclidean distance between the points
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def calculate_mae(true_traj, pred_traj):
    """
    Calculate Mean Absolute Error between true and predicted trajectories.
    
    Args:
        true_traj: List of trajectory points (x, y, z)
        pred_traj: List of predicted trajectory points (x, y, z)
    
    Returns:
        MAE value
    """
    if not true_traj or not pred_traj:
        raise ValueError("Trajectories cannot be empty")
    
    if len(true_traj) != len(pred_traj):
        raise ValueError("Trajectories must have the same length")
    
    total_error = sum(euclidean_dist(true_traj[i], pred_traj[i]) for i in range(len(true_traj)))
    return total_error / len(true_traj)

def calculate_rmse(true_traj, pred_traj):
    """
    Calculate Root Mean Square Error between true and predicted trajectories.
    
    Args:
        true_traj: List of trajectory points (x, y, z)
        pred_traj: List of predicted trajectory points (x, y, z)
    
    Returns:
        RMSE value
    """
    if not true_traj or not pred_traj:
        raise ValueError("Trajectories cannot be empty")
        
    if len(true_traj) != len(pred_traj):
        raise ValueError("Trajectories must have the same length")
    
    squared_error_sum = sum(euclidean_dist(true_traj[i], pred_traj[i])**2 for i in range(len(true_traj)))
    return math.sqrt(squared_error_sum / len(true_traj))


if __name__ == "__main__":
    # Run all tests

    
    # Example usage of the metrics functions
    true_traj = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
    pred_traj = [(0.1, 0.1, 0.1), (1.1, 0.9, 1.1), (2.1, 2.1, 1.9)]
    
    mae = calculate_mae(true_traj, pred_traj)
    rmse = calculate_rmse(true_traj, pred_traj)
    
    print(f"Trajectory Evaluation Metrics:")
    print(f"MAE: {mae:.4f} m")
    print(f"RMSE: {rmse:.4f} m")
