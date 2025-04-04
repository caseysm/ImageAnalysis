"""Main pipeline for automated coordinate mapping between magnifications."""

import numpy as np
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt

from .model import TransformParameters, calculate_transform_error
from .matching import iterative_matching

@dataclass
class MappingResult:
    """Results from mapping between magnifications."""
    transform: TransformParameters
    matched_points_10x: np.ndarray
    matched_points_40x: np.ndarray
    error_metrics: Dict[str, float]
    
    def save(self, output_path: str) -> None:
        """Save mapping results to JSON file."""
        data = {
            'transform': {
                'dx': self.transform.dx,
                'dy': self.transform.dy,
                'theta': self.transform.theta,
                'scale_x': self.transform.scale_x,
                'scale_y': self.transform.scale_y
            },
            'error_metrics': self.error_metrics,
            'num_matched_points': len(self.matched_points_10x)
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def plot_diagnostics(self, output_dir: str) -> None:
        """Generate diagnostic plots for the mapping."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot error histogram
        errors = np.sqrt(np.sum(
            (self.matched_points_40x - self.matched_points_10x)**2,
            axis=1
        ))
        
        plt.figure(figsize=(8, 6))
        plt.hist(errors, bins=50)
        plt.xlabel('Error (pixels)')
        plt.ylabel('Count')
        plt.title('Distribution of Matching Errors')
        plt.savefig(os.path.join(output_dir, 'error_histogram.png'))
        plt.close()
        
        # Plot point correspondences
        plt.figure(figsize=(12, 6))
        
        plt.subplot(121)
        plt.scatter(self.matched_points_10x[:, 0],
                   self.matched_points_10x[:, 1],
                   c=errors, cmap='viridis')
        plt.colorbar(label='Error (pixels)')
        plt.title('10X Points')
        plt.xlabel('X')
        plt.ylabel('Y')
        
        plt.subplot(122)
        plt.scatter(self.matched_points_40x[:, 0],
                   self.matched_points_40x[:, 1],
                   c=errors, cmap='viridis')
        plt.colorbar(label='Error (pixels)')
        plt.title('40X Points')
        plt.xlabel('X')
        plt.ylabel('Y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'point_correspondences.png'))
        plt.close()

class MappingPipeline:
    """Pipeline for automated coordinate mapping between magnifications."""
    
    def __init__(self,
                 seg_10x_dir: str,
                 seg_40x_dir: str,
                 output_dir: str,
                 config: Optional[Dict] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize the mapping pipeline.
        
        Args:
            seg_10x_dir: Directory containing 10X segmentation results
            seg_40x_dir: Directory containing 40X segmentation results
            output_dir: Directory to save mapping results
            config: Optional configuration dictionary
            logger: Optional logger instance
        """
        self.seg_10x_dir = Path(seg_10x_dir)
        self.seg_40x_dir = Path(seg_40x_dir)
        self.output_dir = Path(output_dir)
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_nuclei_centroids(self, well: str, magnification: str) -> np.ndarray:
        """Load nuclei centroids for a specific well and magnification.
        
        Args:
            well: Well identifier
            magnification: Either '10x' or '40x'
            
        Returns:
            Nx2 array of (x, y) coordinates
        """
        if magnification.lower() == '10x':
            seg_dir = self.seg_10x_dir
            
            # First check in the well subdirectory
            well_dir = seg_dir / well
            if well_dir.exists():
                # Check for centroids in the well directory
                centroids_file = well_dir / f"{well}_nuclei_centroids.npy"
                if centroids_file.exists():
                    self.logger.info(f"Loading 10X centroids from {centroids_file}")
                    return np.load(centroids_file)
                
                # Try any centroids file in this directory
                centroids_files = list(well_dir.glob("*_nuclei_centroids*.npy"))
                if centroids_files:
                    self.logger.info(f"Loading 10X centroids from {centroids_files[0]}")
                    return np.load(centroids_files[0])
            
            # If not found in well directory, try the main directory
            centroids_file = seg_dir / f"{well}_nuclei_centroids.npy"
            if centroids_file.exists():
                self.logger.info(f"Loading 10X centroids from {centroids_file}")
                return np.load(centroids_file)
                
            # Try recursive glob to find any matching files
            self.logger.info(f"Searching recursively for 10X centroids in {seg_dir}")
            centroids_files = list(seg_dir.glob(f"**/{well}*_nuclei_centroids*.npy"))
            if centroids_files:
                # Combine all centroids from multiple files
                all_centroids = []
                for f in centroids_files:
                    self.logger.info(f"Loading 10X centroids from {f}")
                    centroids = np.load(f)
                    all_centroids.append(centroids)
                
                if all_centroids:
                    return np.vstack(all_centroids)
            
            # If still not found, raise error
            raise FileNotFoundError(f"Could not find any centroids files for well {well} in {seg_dir}")
        else:
            # For 40X, we use a different naming convention
            seg_dir = self.seg_40x_dir
            
            # First check in the well subdirectory
            well_dir = seg_dir / well
            if well_dir.exists():
                # Check for 40X centroids in the well directory
                centroids_file = well_dir / f"{well}_nuclei_centroids_40x.npy"
                if centroids_file.exists():
                    self.logger.info(f"Loading 40X centroids from {centroids_file}")
                    return np.load(centroids_file)
                
                # Try any centroids file in this directory
                centroids_files = list(well_dir.glob("*_nuclei_centroids*.npy"))
                if centroids_files:
                    self.logger.info(f"Loading 40X centroids from {centroids_files[0]}")
                    return np.load(centroids_files[0])
            
            # If not found in well directory, try the main directory
            centroids_file = seg_dir / f"{well}_nuclei_centroids_40x.npy"
            if centroids_file.exists():
                self.logger.info(f"Loading 40X centroids from {centroids_file}")
                return np.load(centroids_file)
                
            # Try recursive glob to find any matching files
            self.logger.info(f"Searching recursively for 40X centroids in {seg_dir}")
            centroids_files = list(seg_dir.glob(f"**/{well}*_nuclei_centroids*.npy"))
            if centroids_files:
                # Combine all centroids from multiple files
                all_centroids = []
                for f in centroids_files:
                    self.logger.info(f"Loading 40X centroids from {f}")
                    centroids = np.load(f)
                    all_centroids.append(centroids)
                
                if all_centroids:
                    return np.vstack(all_centroids)
            
            # If still not found, raise error
            raise FileNotFoundError(f"Could not find any centroids files for well {well} in {seg_dir}")
        
    def process_well(self, well: str) -> MappingResult:
        """Process a single well to find coordinate mapping.
        
        Args:
            well: Well identifier
            
        Returns:
            MappingResult object containing transformation and diagnostics
        """
        self.logger.info(f"Processing well {well}")
        
        # Load nuclei centroids
        try:
            points_10x = self.load_nuclei_centroids(well, '10x')
            points_40x = self.load_nuclei_centroids(well, '40x')
        except FileNotFoundError as e:
            self.logger.error(f"Failed to load centroids for well {well}: {e}")
            raise
            
        self.logger.info(f"Loaded {len(points_10x)} 10X points and {len(points_40x)} 40X points")
        
        # Get matching parameters from config
        matching_params = self.config.get('matching', {})
        max_iterations = matching_params.get('max_iterations', 5)
        distance_threshold = matching_params.get('distance_threshold', 50.0)
        ransac_threshold = matching_params.get('ransac_threshold', 20.0)
        
        # Perform iterative matching
        try:
            matched_10x, matched_40x, transform = iterative_matching(
                points_10x, points_40x,
                max_iterations=max_iterations,
                distance_threshold=distance_threshold,
                ransac_threshold=ransac_threshold
            )
        except Exception as e:
            self.logger.error(f"Matching failed for well {well}: {e}")
            raise
            
        self.logger.info(f"Found {len(matched_10x)} matched points")
        
        # Calculate error metrics
        error_metrics = calculate_transform_error(matched_10x, matched_40x, transform)
        self.logger.info(f"RMSE: {error_metrics['rmse']:.2f} pixels")
        
        return MappingResult(
            transform=transform,
            matched_points_10x=matched_10x,
            matched_points_40x=matched_40x,
            error_metrics=error_metrics
        )
        
    def run(self, wells: Optional[List[str]] = None) -> Dict[str, MappingResult]:
        """Run the mapping pipeline on specified wells.
        
        Args:
            wells: Optional list of wells to process (default: all wells)
            
        Returns:
            Dictionary mapping well IDs to MappingResults
        """
        # If no wells specified, try to find all wells
        if wells is None:
            wells = self._find_available_wells()
            
        results = {}
        for well in wells:
            try:
                result = self.process_well(well)
                
                # Save results
                result.save(self.output_dir / f"{well}_mapping.json")
                result.plot_diagnostics(self.output_dir / f"{well}_diagnostics")
                
                results[well] = result
                
            except Exception as e:
                self.logger.error(f"Failed to process well {well}: {e}")
                continue
                
        return results
        
    def _find_available_wells(self) -> List[str]:
        """Find all wells that have both 10X and 40X data available."""
        # Look for all centroids files with various naming patterns, including in subdirectories
        wells_10x = set()
        
        # First check the well subdirectories
        for well_dir in self.seg_10x_dir.glob("Well*"):
            if well_dir.is_dir():
                well_id = well_dir.name
                
                # Check if well directory contains any centroids files
                centroids_files = list(well_dir.glob("*_nuclei_centroids*.npy"))
                if centroids_files:
                    self.logger.info(f"Found 10X centroids in {well_dir}: {centroids_files}")
                    wells_10x.add(well_id)
        
        # Check for centroids files directly in the main directory
        for pattern in ["*_nuclei_centroids.npy", "*_nuclei_centroids_*.npy"]:
            for f in self.seg_10x_dir.glob(pattern):
                # Extract well ID from filename (assuming Well1_...)
                well_id = f.stem.split('_')[0]
                if well_id.startswith("Well"):
                    self.logger.info(f"Found 10X centroids file in main directory: {f}")
                    wells_10x.add(well_id)
        
        # Do the same for 40X
        wells_40x = set()
        
        # First check the well subdirectories
        for well_dir in self.seg_40x_dir.glob("Well*"):
            if well_dir.is_dir():
                well_id = well_dir.name
                
                # Check if well directory contains any centroids files
                centroids_files = list(well_dir.glob("*_nuclei_centroids*.npy"))
                if centroids_files:
                    self.logger.info(f"Found 40X centroids in {well_dir}: {centroids_files}")
                    wells_40x.add(well_id)
        
        # Check for centroids files directly in the main directory
        for pattern in ["*_nuclei_centroids_40x.npy", "*_nuclei_centroids_*.npy"]:
            for f in self.seg_40x_dir.glob(pattern):
                # Extract well ID from filename
                well_id = f.stem.split('_')[0]
                if well_id.startswith("Well"):
                    self.logger.info(f"Found 40X centroids file in main directory: {f}")
                    wells_40x.add(well_id)
        
        self.logger.info(f"Found {len(wells_10x)} wells with 10X centroids: {wells_10x}")
        self.logger.info(f"Found {len(wells_40x)} wells with 40X centroids: {wells_40x}")
        
        # Get intersection of wells
        available_wells = sorted(wells_10x & wells_40x)
        self.logger.info(f"Found {len(available_wells)} wells with both 10X and 40X data: {available_wells}")
        
        return available_wells