"""Module containing functions for calculating cellular phenotype metrics."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from skimage import measure, feature, filters
from scipy import ndimage
import mahotas as mh

def calculate_area_metrics(
    mask: np.ndarray,
    pixel_size: float = 1.0
) -> Dict[str, float]:
    """Calculate area-based metrics for a cell.
    
    Args:
        mask: Binary mask of the cell
        pixel_size: Physical size of each pixel in microns
        
    Returns:
        Dictionary containing area metrics:
            - area: Total area in square microns
            - perimeter: Perimeter length in microns
            - equivalent_diameter: Diameter of circle with same area
            - major_axis_length: Length of major axis
            - minor_axis_length: Length of minor axis
    """
    # Get region properties
    props = measure.regionprops(mask.astype(int))[0]
    
    # Calculate metrics
    metrics = {
        'area': props.area * (pixel_size ** 2),
        'perimeter': props.perimeter * pixel_size,
        'equivalent_diameter': props.equivalent_diameter * pixel_size,
        'major_axis_length': props.major_axis_length * pixel_size,
        'minor_axis_length': props.minor_axis_length * pixel_size
    }
    
    return metrics

def calculate_shape_metrics(
    mask: np.ndarray
) -> Dict[str, float]:
    """Calculate shape metrics for a cell.
    
    Args:
        mask: Binary mask of the cell
        
    Returns:
        Dictionary containing shape metrics:
            - circularity: 4π*area/perimeter^2 (1 for perfect circle)
            - eccentricity: Eccentricity of ellipse fit
            - solidity: Ratio of area to convex hull area
            - extent: Ratio of area to bounding box area
            - orientation: Angle of major axis with x-axis
    """
    # Get region properties
    props = measure.regionprops(mask.astype(int))[0]
    
    # Calculate circularity (1 for perfect circle)
    area = props.area
    perimeter = props.perimeter
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    
    metrics = {
        'circularity': circularity,
        'eccentricity': props.eccentricity,
        'solidity': props.solidity,
        'extent': props.extent,
        'orientation': props.orientation
    }
    
    return metrics

def calculate_intensity_metrics(
    mask: np.ndarray,
    image: np.ndarray,
    background: Optional[float] = None
) -> Dict[str, float]:
    """Calculate intensity metrics for a cell in a given channel.
    
    Args:
        mask: Binary mask of the cell
        image: Intensity image (single channel)
        background: Optional background intensity to subtract
        
    Returns:
        Dictionary containing intensity metrics:
            - mean_intensity: Mean intensity within cell
            - integrated_intensity: Total intensity within cell
            - max_intensity: Maximum intensity within cell
            - min_intensity: Minimum intensity within cell
            - std_intensity: Standard deviation of intensity
    """
    # Apply mask
    masked_image = image * mask
    
    # Get intensities within mask
    intensities = masked_image[mask > 0]
    
    # Subtract background if provided
    if background is not None:
        intensities = intensities - background
        
    metrics = {
        'mean_intensity': np.mean(intensities),
        'integrated_intensity': np.sum(intensities),
        'max_intensity': np.max(intensities),
        'min_intensity': np.min(intensities),
        'std_intensity': np.std(intensities)
    }
    
    return metrics

def calculate_texture_metrics(
    mask: np.ndarray,
    image: np.ndarray,
    distances: List[int] = [1, 2, 4]
) -> Dict[str, float]:
    """Calculate texture metrics using Haralick features.
    
    Args:
        mask: Binary mask of the cell
        image: Intensity image (single channel)
        distances: List of pixel distances for GLCM calculation
        
    Returns:
        Dictionary containing texture metrics:
            - contrast: Local intensity variation
            - correlation: Linear dependency of gray levels
            - energy: Sum of squared GLCM elements
            - homogeneity: Closeness of GLCM elements to diagonal
            - entropy: Randomness of gray level distribution
    """
    # Apply mask and normalize to 8-bit range
    masked_image = image * mask
    masked_image = ((masked_image - np.min(masked_image)) * 255 / 
                   (np.max(masked_image) - np.min(masked_image))).astype(np.uint8)
    
    # Calculate Haralick features
    haralick_features = mh.features.haralick(
        masked_image,
        ignore_zeros=True,
        distance=distances,
        return_mean=True
    )
    
    metrics = {
        'contrast': haralick_features[1],
        'correlation': haralick_features[2],
        'energy': haralick_features[8],
        'homogeneity': haralick_features[4],
        'entropy': haralick_features[9]
    }
    
    return metrics

def calculate_location_metrics(
    mask: np.ndarray,
    well_center: Optional[Tuple[float, float]] = None,
    pixel_size: float = 1.0
) -> Dict[str, float]:
    """Calculate location and spatial metrics for a cell.
    
    Args:
        mask: Binary mask of the cell
        well_center: Optional (y, x) coordinates of well center
        pixel_size: Physical size of each pixel in microns
        
    Returns:
        Dictionary containing location metrics:
            - centroid_y: Y-coordinate of centroid
            - centroid_x: X-coordinate of centroid
            - distance_to_center: Distance to well center if provided
            - boundary_distance: Distance to nearest image boundary
    """
    # Get region properties
    props = measure.regionprops(mask.astype(int))[0]
    
    # Get centroid coordinates
    y, x = props.centroid
    metrics = {
        'centroid_y': y * pixel_size,
        'centroid_x': x * pixel_size
    }
    
    # Calculate distance to well center if provided
    if well_center is not None:
        center_y, center_x = well_center
        distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        metrics['distance_to_center'] = distance * pixel_size
        
    # Calculate distance to nearest boundary
    y_size, x_size = mask.shape
    boundary_dist = min(y, x, y_size - y, x_size - x)
    metrics['boundary_distance'] = boundary_dist * pixel_size
    
    return metrics

def calculate_all_metrics(
    cell_mask: np.ndarray,
    nuclei_mask: np.ndarray,
    channels: Dict[str, np.ndarray],
    pixel_size: float = 1.0,
    well_center: Optional[Tuple[float, float]] = None,
    background_intensities: Optional[Dict[str, float]] = None
) -> Dict[str, Dict[str, float]]:
    """Calculate all phenotype metrics for a cell.
    
    Args:
        cell_mask: Binary mask of the cell
        nuclei_mask: Binary mask of the nucleus
        channels: Dictionary mapping channel names to intensity images
        pixel_size: Physical size of each pixel in microns
        well_center: Optional (y, x) coordinates of well center
        background_intensities: Optional dict of background intensities per channel
        
    Returns:
        Nested dictionary containing all metrics organized by category
    """
    metrics = {}
    
    # Calculate area metrics for both cell and nucleus
    metrics['cell_area'] = calculate_area_metrics(cell_mask, pixel_size)
    metrics['nuclear_area'] = calculate_area_metrics(nuclei_mask, pixel_size)
    
    # Calculate shape metrics
    metrics['cell_shape'] = calculate_shape_metrics(cell_mask)
    metrics['nuclear_shape'] = calculate_shape_metrics(nuclei_mask)
    
    # Calculate intensity metrics for each channel
    metrics['intensity'] = {}
    for channel_name, image in channels.items():
        background = (background_intensities.get(channel_name) 
                     if background_intensities else None)
        
        # Calculate for both cell and nucleus
        cell_intensities = calculate_intensity_metrics(
            cell_mask, image, background
        )
        nuclear_intensities = calculate_intensity_metrics(
            nuclei_mask, image, background
        )
        
        metrics['intensity'][channel_name] = {
            'cell': cell_intensities,
            'nucleus': nuclear_intensities
        }
        
        # Calculate texture metrics
        metrics['texture'] = {
            channel_name: calculate_texture_metrics(cell_mask, image)
        }
    
    # Calculate location metrics
    metrics['location'] = calculate_location_metrics(
        cell_mask,
        well_center,
        pixel_size
    )
    
    return metrics 