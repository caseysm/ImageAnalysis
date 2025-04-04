#!/usr/bin/env python3
"""Test script for validating segmentation pipeline against original implementation."""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directories to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.segmentation.segmentation_10x import Segmentation10XPipeline
from core.segmentation.segmentation_40x import Segmentation40XPipeline

def compare_segmentation_results(original_output, new_output, tolerance=0.01):
    """Compare segmentation results between original and new implementations.
    
    Args:
        original_output: Path to original output directory
        new_output: Path to new output directory
        tolerance: Allowed difference in numerical values
        
    Returns:
        Dictionary of comparison results
    """
    results = {
        'match': True,
        'differences': [],
        'missing_files': [],
        'extra_files': []
    }
    
    # Check for cell masks
    orig_masks = sorted(list(Path(original_output).glob('*_cell_masks.npy')))
    new_masks = sorted(list(Path(new_output).glob('*_cell_masks.npy')))
    
    # Check for missing files
    orig_mask_names = {os.path.basename(m) for m in orig_masks}
    new_mask_names = {os.path.basename(m) for m in new_masks}
    results['missing_files'] = list(orig_mask_names - new_mask_names)
    results['extra_files'] = list(new_mask_names - orig_mask_names)
    
    # Compare common masks
    common_masks = orig_mask_names.intersection(new_mask_names)
    for mask_name in common_masks:
        orig_path = Path(original_output) / mask_name
        new_path = Path(new_output) / mask_name
        
        orig_data = np.load(orig_path)
        new_data = np.load(new_path)
        
        # Check shape
        if orig_data.shape != new_data.shape:
            results['match'] = False
            results['differences'].append(f"{mask_name}: Shape mismatch - "
                                         f"original {orig_data.shape} vs new {new_data.shape}")
            continue
            
        # Check cell count (number of unique non-zero values)
        orig_cells = len(np.unique(orig_data)) - 1  # Subtract 1 for background
        new_cells = len(np.unique(new_data)) - 1
        
        if abs(orig_cells - new_cells) > orig_cells * tolerance:
            results['match'] = False
            results['differences'].append(f"{mask_name}: Cell count mismatch - "
                                         f"original {orig_cells} vs new {new_cells}")
            
        # Check overlap (IoU) for each cell - this would require more complex matching
        # Simplified here to just check cell areas
        orig_areas = pd.Series(np.bincount(orig_data.flatten())[1:])  # Skip background
        new_areas = pd.Series(np.bincount(new_data.flatten())[1:])
        
        if len(orig_areas) > 0 and len(new_areas) > 0:
            orig_mean_area = orig_areas.mean()
            new_mean_area = new_areas.mean()
            
            if abs(orig_mean_area - new_mean_area) / orig_mean_area > tolerance:
                results['match'] = False
                results['differences'].append(f"{mask_name}: Mean cell area mismatch - "
                                             f"original {orig_mean_area:.2f} vs new {new_mean_area:.2f}")
    
    return results

def run_test(args):
    """Run the segmentation test.
    
    Args:
        args: Command-line arguments
    """
    # Create test output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Select pipeline based on magnification
    if args.magnification == '10x':
        pipeline_class = Segmentation10XPipeline
    else:
        pipeline_class = Segmentation40XPipeline
    
    # Create pipeline
    pipeline = pipeline_class(
        input_file=args.input_file,
        output_dir=args.output_dir,
    )
    
    # Run pipeline with a single well
    pipeline.run(wells=[args.well])
    
    # Compare results if original results are available
    if args.original_output:
        results = compare_segmentation_results(args.original_output, args.output_dir)
        
        print("\n=== Segmentation Comparison Results ===")
        print(f"Overall match: {results['match']}")
        
        if results['missing_files']:
            print("\nMissing files in new implementation:")
            for f in results['missing_files']:
                print(f"  - {f}")
                
        if results['extra_files']:
            print("\nExtra files in new implementation:")
            for f in results['extra_files']:
                print(f"  - {f}")
                
        if results['differences']:
            print("\nDifferences found:")
            for diff in results['differences']:
                print(f"  - {diff}")
        
        if results['match']:
            print("\nSUCCESS: New implementation matches original.")
        else:
            print("\nWARNING: Differences found between implementations.")
    
def parse_args():
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Test segmentation pipeline against original implementation"
    )
    
    parser.add_argument(
        '--input-file',
        required=True,
        help="Path to input ND2 file"
    )
    
    parser.add_argument(
        '--well',
        default='Well1',
        help="Well to process (default: Well1)"
    )
    
    parser.add_argument(
        '--magnification',
        choices=['10x', '40x'],
        default='10x',
        help="Microscope magnification (default: 10x)"
    )
    
    parser.add_argument(
        '--output-dir',
        default='./results/test_segmentation',
        help="Output directory (default: ./results/test_segmentation)"
    )
    
    parser.add_argument(
        '--original-output',
        help="Path to original implementation output for comparison"
    )
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run_test(args)