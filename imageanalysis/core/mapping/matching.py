"""Functions for matching nuclei between magnifications and filtering matches."""

import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, List, Optional
import random
from .model import TransformParameters, apply_transform, optimize_lbfgs

def find_nearest_neighbors(points_10x: np.ndarray, points_40x: np.ndarray,
                         max_distance: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
    """Find nearest neighbor matches between 10X and 40X points.
    
    Args:
        points_10x: Nx2 array of 10X coordinates
        points_40x: Mx2 array of 40X coordinates
        max_distance: Maximum allowed distance for matches
        
    Returns:
        Tuple of (matched_10x, matched_40x) coordinates
    """
    # Build KD-tree for efficient nearest neighbor search
    tree = cKDTree(points_40x)
    
    # Find nearest neighbors and distances
    distances, indices = tree.query(points_10x, k=1)
    
    # Filter matches by distance threshold
    valid_matches = distances <= max_distance
    matched_10x = points_10x[valid_matches]
    matched_40x = points_40x[indices[valid_matches]]
    
    return matched_10x, matched_40x

def ransac_filter(points_10x: np.ndarray, points_40x: np.ndarray,
                 num_iterations: int = 200, sample_size: int = 4,  # Even more reduced sample size
                 inlier_threshold: float = 30.0,  # More lenient threshold
                 min_inliers_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:  # Much more lenient ratio
    """Filter point matches using RANSAC algorithm.
    
    Args:
        points_10x: Nx2 array of 10X coordinates
        points_40x: Nx2 array of corresponding 40X coordinates
        num_iterations: Number of RANSAC iterations
        sample_size: Number of point pairs to sample in each iteration
        inlier_threshold: Maximum distance for inlier classification
        min_inliers_ratio: Minimum ratio of inliers to consider a good model
        
    Returns:
        Tuple of (filtered_10x, filtered_40x) containing only inlier matches
    """
    assert points_10x.shape == points_40x.shape
    n_points = len(points_10x)
    
    best_inliers = None
    best_num_inliers = 0
    
    for _ in range(num_iterations):
        # Randomly sample point pairs
        sample_indices = random.sample(range(n_points), sample_size)
        sample_10x = points_10x[sample_indices]
        sample_40x = points_40x[sample_indices]
        
        # Estimate transformation from samples
        try:
            transform = optimize_lbfgs(sample_10x, sample_40x)
        except:
            continue
            
        # Transform all points and calculate errors
        transformed = apply_transform(points_10x, transform)
        errors = np.sqrt(np.sum((transformed - points_40x)**2, axis=1))
        
        # Find inliers
        inliers = errors <= inlier_threshold
        num_inliers = np.sum(inliers)
        
        # Update best model if we found more inliers
        if num_inliers > best_num_inliers:
            best_inliers = inliers
            best_num_inliers = num_inliers
    
    # Check if we found a good model
    if best_inliers is None or best_num_inliers < min_inliers_ratio * n_points:
        raise ValueError("RANSAC failed to find a good model")
    
    return points_10x[best_inliers], points_40x[best_inliers]

def iterative_matching(points_10x: np.ndarray, points_40x: np.ndarray,
                      initial_transform: Optional[TransformParameters] = None,
                      max_iterations: int = 5,
                      distance_threshold: float = 50.0,
                      ransac_threshold: float = 20.0) -> Tuple[np.ndarray, np.ndarray, TransformParameters]:
    """Perform iterative matching between 10X and 40X points.
    
    Args:
        points_10x: Nx2 array of 10X coordinates
        points_40x: Mx2 array of 40X coordinates
        initial_transform: Optional initial transformation guess
        max_iterations: Maximum number of refinement iterations
        distance_threshold: Maximum distance for nearest neighbor matching
        ransac_threshold: Maximum distance for RANSAC inlier classification
        
    Returns:
        Tuple of (matched_10x, matched_40x, final_transform)
    """
    current_transform = initial_transform
    
    for iteration in range(max_iterations):
        # Transform 10X points if we have a current transform
        if current_transform is not None:
            transformed_10x = apply_transform(points_10x, current_transform)
        else:
            transformed_10x = points_10x
            
        # Find nearest neighbor matches
        matched_10x, matched_40x = find_nearest_neighbors(
            transformed_10x, points_40x,
            max_distance=distance_threshold
        )
        
        if len(matched_10x) < 5:  # Require fewer points
            raise ValueError(f"Too few matches found: {len(matched_10x)}")
            
        # Filter matches using RANSAC
        matched_10x, matched_40x = ransac_filter(
            matched_10x, matched_40x,
            inlier_threshold=ransac_threshold
        )
        
        # Optimize transformation
        try:
            current_transform = optimize_lbfgs(matched_10x, matched_40x)
        except:
            break
            
    if current_transform is None:
        raise ValueError("Failed to find valid transformation")
        
    return matched_10x, matched_40x, current_transform 