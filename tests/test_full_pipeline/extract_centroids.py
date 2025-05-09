#!/usr/bin/env python3
"""Script to extract centroids from segmentation masks for the full pipeline test."""

import os
import sys
import numpy as np
from pathlib import Path
import re
from skimage.measure import regionprops

def is_10x_file(filename):
    """Determine if a file is from 10X or 40X data based on its name."""
    match = re.search(r'Seq(\d+)', filename)
    if match:
        seq_num = int(match.group(1))
        return seq_num < 700
    return False

def extract_centroids_from_mask(mask_file):
    """Extract centroids from a nuclei mask file."""
    mask = np.load(mask_file)
    props = regionprops(mask)
    return np.array([prop.centroid for prop in props])

def extract_and_save_centroids(seg_dir, output_dir):
    """Extract centroids from all nuclei masks and save them."""
    # Get all well directories
    well_dirs = [d for d in Path(seg_dir).glob("Well*") if d.is_dir()]
    
    for well_dir in well_dirs:
        well_id = well_dir.name
        print(f"Processing well: {well_id}")
        
        # Create output directories
        well_output_dir = Path(output_dir) / well_id
        os.makedirs(well_output_dir, exist_ok=True)
        
        # Get all nuclei mask files
        nuclei_files = list(well_dir.glob("*_nuclei_mask.npy"))
        
        # Group files by magnification
        files_10x = []
        files_40x = []
        
        for file in nuclei_files:
            if is_10x_file(file.name):
                files_10x.append(file)
            else:
                files_40x.append(file)
        
        print(f"Found {len(files_10x)} 10X files and {len(files_40x)} 40X files")
        
        # Extract centroids for 10X
        centroids_10x = []
        for file in files_10x:
            try:
                centroid = extract_centroids_from_mask(file)
                centroids_10x.append(centroid)
                print(f"Extracted {len(centroid)} centroids from {file.name}")
            except Exception as e:
                print(f"Error extracting centroids from {file.name}: {e}")
        
        # Combine all 10X centroids
        if centroids_10x:
            all_centroids_10x = np.vstack(centroids_10x)
            np.save(well_output_dir / f"{well_id}_nuclei_centroids.npy", all_centroids_10x)
            print(f"Saved {len(all_centroids_10x)} 10X centroids")
        
        # Extract centroids for 40X
        centroids_40x = []
        for file in files_40x:
            try:
                centroid = extract_centroids_from_mask(file)
                centroids_40x.append(centroid)
                print(f"Extracted {len(centroid)} centroids from {file.name}")
            except Exception as e:
                print(f"Error extracting centroids from {file.name}: {e}")
        
        # Combine all 40X centroids
        if centroids_40x:
            all_centroids_40x = np.vstack(centroids_40x)
            np.save(well_output_dir / f"{well_id}_nuclei_centroids_40x.npy", all_centroids_40x)
            print(f"Saved {len(all_centroids_40x)} 40X centroids")

def main():
    """Main function."""
    if len(sys.argv) < 3:
        print("Usage: python extract_centroids.py SEG_DIR OUTPUT_DIR")
        return 1
    
    seg_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.exists(seg_dir):
        print(f"Segmentation directory not found: {seg_dir}")
        return 1
    
    os.makedirs(output_dir, exist_ok=True)
    
    extract_and_save_centroids(seg_dir, output_dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())