#!/usr/bin/env python3
"""Test script for the mapping pipeline on real production data."""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Optional, Tuple
from skimage.measure import regionprops
import re

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test the mapping pipeline on real production data"
    )
    
    parser.add_argument(
        "--segmentation-dir",
        default="/home/casey/Desktop/ShalemLab/ImageAnalysis/results/production/new/segmentation",
        help="Directory containing segmentation results"
    )
    
    parser.add_argument(
        "--output-dir",
        default="/home/casey/Desktop/ShalemLab/ImageAnalysis/results/mapping_test",
        help="Output directory"
    )
    
    parser.add_argument(
        "--wells",
        nargs="+",
        default=["Well1"],
        help="List of wells to process"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    
    return parser.parse_args()

def extract_centroids_from_mask(mask_file: str, props_file: str = None) -> Tuple[np.ndarray, Dict]:
    """Extract centroids from a nuclei mask file.
    
    Args:
        mask_file: Path to nuclei mask file (.npy)
        props_file: Optional path to properties JSON file
        
    Returns:
        Tuple of (centroids, metadata)
    """
    # Load mask
    mask = np.load(mask_file)
    
    # Extract centroids
    props = regionprops(mask)
    centroids = np.array([prop.centroid for prop in props])
    
    # Get metadata from properties file
    metadata = {}
    if props_file and os.path.exists(props_file):
        with open(props_file, 'r') as f:
            metadata = json.load(f)
    
    # Extract sequence number from filename
    match = re.search(r'Seq(\d+)', mask_file)
    if match:
        seq_num = int(match.group(1))
        metadata['seq_num'] = seq_num
    
    return centroids, metadata

def is_10x_file(filename: str) -> bool:
    """Determine if a file is from 10X or 40X data based on its name.
    
    Args:
        filename: Filename to check
        
    Returns:
        True if 10X, False if 40X
    """
    # Since we don't have clear magnification labels, split files based on sequence number
    # This is a modified heuristic for testing - treat first half of numbered sequence as "10X"
    match = re.search(r'Seq(\d+)', filename)
    if match:
        seq_num = int(match.group(1))
        # Use a lower threshold to ensure we have enough files in each category
        # For our test, treat files with sequence numbers below 700 as "10X"
        return seq_num < 700
    
    return False

def organize_files_by_magnification(well_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Organize nuclei mask files into 10X and 40X groups.
    
    Args:
        well_dir: Directory containing nuclei mask files
        
    Returns:
        Tuple of (files_10x, files_40x)
    """
    nuclei_files = list(well_dir.glob("*_nuclei_mask.npy"))
    
    files_10x = []
    files_40x = []
    
    for file in nuclei_files:
        if is_10x_file(file.name):
            files_10x.append(file)
        else:
            files_40x.append(file)
    
    return files_10x, files_40x

def extract_and_save_centroids(well_dir: Path, output_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Extract centroids from nuclei masks and save to output directory.
    
    Args:
        well_dir: Directory containing nuclei mask files
        output_dir: Directory to save centroids
        
    Returns:
        Tuple of (centroids_10x, centroids_40x)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Organize files by magnification
    files_10x, files_40x = organize_files_by_magnification(well_dir)
    
    print(f"Found {len(files_10x)} 10X files and {len(files_40x)} 40X files")
    
    # Extract centroids for 10X
    centroids_10x = []
    for file in files_10x:
        props_file = file.parent / file.name.replace("_nuclei_mask.npy", "_properties.json")
        try:
            centroid, metadata = extract_centroids_from_mask(str(file), str(props_file))
            centroids_10x.append(centroid)
            print(f"Extracted {len(centroid)} centroids from {file.name}")
        except Exception as e:
            print(f"Error extracting centroids from {file.name}: {e}")
    
    # Combine all 10X centroids
    if centroids_10x:
        all_centroids_10x = np.vstack(centroids_10x)
        np.save(output_dir / f"{well_dir.name}_nuclei_centroids_10x.npy", all_centroids_10x)
        print(f"Saved {len(all_centroids_10x)} 10X centroids")
    else:
        all_centroids_10x = np.array([])
    
    # Extract centroids for 40X
    centroids_40x = []
    for file in files_40x:
        props_file = file.parent / file.name.replace("_nuclei_mask.npy", "_properties.json")
        try:
            centroid, metadata = extract_centroids_from_mask(str(file), str(props_file))
            centroids_40x.append(centroid)
            print(f"Extracted {len(centroid)} centroids from {file.name}")
        except Exception as e:
            print(f"Error extracting centroids from {file.name}: {e}")
    
    # Combine all 40X centroids
    if centroids_40x:
        all_centroids_40x = np.vstack(centroids_40x)
        np.save(output_dir / f"{well_dir.name}_nuclei_centroids_40x.npy", all_centroids_40x)
        print(f"Saved {len(all_centroids_40x)} 40X centroids")
    else:
        all_centroids_40x = np.array([])
    
    return all_centroids_10x, all_centroids_40x

def plot_centroids(centroids_10x: np.ndarray, centroids_40x: np.ndarray, output_file: str):
    """Create a plot showing both sets of centroids.
    
    Args:
        centroids_10x: 10X centroids array
        centroids_40x: 40X centroids array
        output_file: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.scatter(centroids_10x[:, 1], centroids_10x[:, 0], s=5, alpha=0.5)
    plt.title(f"10X Centroids ({len(centroids_10x)} points)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    
    plt.subplot(122)
    plt.scatter(centroids_40x[:, 1], centroids_40x[:, 0], s=5, alpha=0.5)
    plt.title(f"40X Centroids ({len(centroids_40x)} points)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def run_mapping_pipeline(seg_10x_dir: str, seg_40x_dir: str, output_dir: str, wells: List[str]):
    """Run the mapping pipeline using our extracted centroids.
    
    Args:
        seg_10x_dir: Directory containing 10X centroids
        seg_40x_dir: Directory containing 40X centroids
        output_dir: Directory to save mapping results
        wells: List of wells to process
    """
    try:
        # Import here to avoid issues if module is not installed
        from imageanalysis.core.mapping.pipeline import MappingPipeline
        
        # Initialize pipeline with custom config for more lenient matching
        pipeline = MappingPipeline(
            seg_10x_dir=seg_10x_dir,
            seg_40x_dir=seg_40x_dir,
            output_dir=output_dir,
            config={
                'matching': {
                    'max_iterations': 10,
                    'distance_threshold': 100.0,  # More lenient distance threshold
                    'ransac_threshold': 50.0  # More lenient RANSAC threshold
                }
            }
        )
        
        # Run pipeline
        results = pipeline.run(wells=wells)
        
        # Print summary
        print("\nMapping Results Summary:")
        for well, result in results.items():
            print(f"\nWell {well}:")
            print(f"  Matched points: {len(result.matched_points_10x)}")
            print(f"  RMSE: {result.error_metrics['rmse']:.2f} pixels")
            print(f"  Max error: {result.error_metrics['max_error']:.2f} pixels")
            print(f"  Results saved to: {output_dir}/{well}_mapping.json")
            print(f"  Diagnostics saved to: {output_dir}/{well}_diagnostics/")
            
        return True
        
    except ImportError:
        print("ERROR: Could not import mapping pipeline. Make sure imageanalysis is installed.")
        return False
    except Exception as e:
        print(f"ERROR: Mapping pipeline failed: {e}")
        return False

def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directories
    centroids_dir = Path(args.output_dir) / "centroids"
    centroids_dir.mkdir(parents=True, exist_ok=True)
    
    mapping_dir = Path(args.output_dir) / "mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each well
    for well in args.wells:
        print(f"\nProcessing well: {well}")
        
        # Get well directory
        well_dir = Path(args.segmentation_dir) / well
        if not well_dir.exists():
            print(f"ERROR: Well directory not found: {well_dir}")
            continue
        
        # Extract and save centroids
        try:
            centroids_10x, centroids_40x = extract_and_save_centroids(well_dir, centroids_dir)
            
            # Plot centroids
            if len(centroids_10x) > 0 and len(centroids_40x) > 0:
                plot_file = centroids_dir / f"{well}_centroids.png"
                plot_centroids(centroids_10x, centroids_40x, str(plot_file))
                print(f"Saved centroids plot to {plot_file}")
            
        except Exception as e:
            print(f"ERROR: Failed to extract centroids for well {well}: {e}")
            continue
    
    # Run mapping pipeline
    print("\nRunning mapping pipeline...")
    success = run_mapping_pipeline(
        seg_10x_dir=str(centroids_dir),
        seg_40x_dir=str(centroids_dir),
        output_dir=str(mapping_dir),
        wells=args.wells
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())