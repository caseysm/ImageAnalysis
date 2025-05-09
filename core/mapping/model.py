"""Transformation model and optimization functions for automated mapping."""

import numpy as np
from scipy.optimize import minimize, least_squares
from typing import Tuple, Dict, Optional, List, Union
from dataclasses import dataclass

@dataclass
class TransformParameters:
    """Parameters for the 5-DOF transformation model."""
    dx: float  # X translation
    dy: float  # Y translation
    theta: float  # Rotation angle in radians
    scale_x: float  # X scaling factor
    scale_y: float  # Y scaling factor

    def to_array(self) -> np.ndarray:
        """Convert parameters to numpy array."""
        return np.array([self.dx, self.dy, self.theta, self.scale_x, self.scale_y])

    @classmethod
    def from_array(cls, params: np.ndarray) -> 'TransformParameters':
        """Create TransformParameters from numpy array."""
        return cls(dx=params[0], dy=params[1], theta=params[2],
                  scale_x=params[3], scale_y=params[4])

def apply_transform(points: np.ndarray, params: TransformParameters) -> np.ndarray:
    """Apply 5-DOF transformation to points.
    
    Args:
        points: Nx2 array of (x, y) coordinates
        params: TransformParameters object
        
    Returns:
        Nx2 array of transformed coordinates
    """
    # Extract parameters
    dx, dy = params.dx, params.dy
    theta = params.theta
    scale_x, scale_y = params.scale_x, params.scale_y
    
    # Create rotation matrix
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Apply transformation
    x = points[:, 0]
    y = points[:, 1]
    
    x_new = scale_x * (cos_theta * x - sin_theta * y + dx)
    y_new = scale_y * (sin_theta * x + cos_theta * y + dy)
    
    return np.column_stack([x_new, y_new])

def calculate_residuals(points_10x: np.ndarray, points_40x: np.ndarray,
                       params: Union[np.ndarray, TransformParameters]) -> np.ndarray:
    """Calculate residuals between transformed 10X points and 40X points.
    
    Args:
        points_10x: Nx2 array of 10X coordinates
        points_40x: Nx2 array of corresponding 40X coordinates
        params: Transform parameters as array or TransformParameters object
        
    Returns:
        2N array of residuals (dx1, dy1, dx2, dy2, ...)
    """
    if isinstance(params, np.ndarray):
        params = TransformParameters.from_array(params)
    
    transformed = apply_transform(points_10x, params)
    return (transformed - points_40x).flatten()

def optimize_lbfgs(points_10x: np.ndarray, points_40x: np.ndarray,
                  initial_params: Optional[TransformParameters] = None,
                  bounds: Optional[List[Tuple[float, float]]] = None) -> TransformParameters:
    """Optimize transformation using L-BFGS-B algorithm.
    
    Args:
        points_10x: Nx2 array of 10X coordinates
        points_40x: Nx2 array of corresponding 40X coordinates
        initial_params: Optional initial parameters (default uses reasonable guesses)
        bounds: Optional parameter bounds as list of (min, max) tuples
        
    Returns:
        Optimized TransformParameters
    """
    if initial_params is None:
        # Estimate initial parameters
        initial_params = TransformParameters(
            dx=0.0,
            dy=0.0,
            theta=0.0,
            scale_x=4.0,
            scale_y=4.0
        )
    
    if bounds is None:
        bounds = [
            (None, None),  # dx
            (None, None),  # dy
            (-np.pi/18, np.pi/18),  # theta (-10 to 10 degrees)
            (3.5, 4.5),  # scale_x
            (3.5, 4.5)   # scale_y
        ]
    
    def objective(params_array: np.ndarray) -> float:
        """Objective function for optimization."""
        residuals = calculate_residuals(points_10x, points_40x, params_array)
        return np.sum(residuals**2)
    
    result = minimize(
        objective,
        initial_params.to_array(),
        method='L-BFGS-B',
        bounds=bounds,
        options={
            'ftol': 1e-8,
            'maxiter': 100
        }
    )
    
    return TransformParameters.from_array(result.x)

def optimize_lm(points_10x: np.ndarray, points_40x: np.ndarray,
               initial_params: Optional[TransformParameters] = None) -> TransformParameters:
    """Optimize transformation using Levenberg-Marquardt algorithm.
    
    Args:
        points_10x: Nx2 array of 10X coordinates
        points_40x: Nx2 array of corresponding 40X coordinates
        initial_params: Optional initial parameters (default uses reasonable guesses)
        
    Returns:
        Optimized TransformParameters
    """
    if initial_params is None:
        initial_params = TransformParameters(
            dx=0.0,
            dy=0.0,
            theta=0.0,
            scale_x=4.0,
            scale_y=4.0
        )
    
    def residual_func(params_array: np.ndarray) -> np.ndarray:
        """Residual function for Levenberg-Marquardt."""
        return calculate_residuals(points_10x, points_40x, params_array)
    
    result = least_squares(
        residual_func,
        initial_params.to_array(),
        method='lm',
        ftol=1e-8,
        xtol=1e-8,
        max_nfev=100
    )
    
    return TransformParameters.from_array(result.x)

def calculate_transform_error(points_10x: np.ndarray, points_40x: np.ndarray,
                            params: TransformParameters) -> Dict[str, float]:
    """Calculate error metrics for the transformation.
    
    Args:
        points_10x: Nx2 array of 10X coordinates
        points_40x: Nx2 array of corresponding 40X coordinates
        params: TransformParameters to evaluate
        
    Returns:
        Dictionary containing error metrics:
            - mean_error: Mean Euclidean distance between matched points
            - median_error: Median Euclidean distance
            - max_error: Maximum Euclidean distance
            - rmse: Root mean square error
    """
    transformed = apply_transform(points_10x, params)
    errors = np.sqrt(np.sum((transformed - points_40x)**2, axis=1))
    
    return {
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'max_error': np.max(errors),
        'rmse': np.sqrt(np.mean(errors**2))
    } 