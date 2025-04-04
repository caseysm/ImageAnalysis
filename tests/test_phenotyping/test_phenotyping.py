#!/usr/bin/env python3
"""Test script for validating phenotyping pipeline against original implementation."""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directories to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.phenotyping.pipeline import StandardPhenotypingPipeline

def compare_phenotyping_results(original_output, new_output, tolerance=0.05):
    """Compare phenotyping results between original and new implementations.
    
    Args:
        original_output: Path to original output directory
        new_output: Path to new output directory
        tolerance: Allowed difference in numerical values (percentage)
        
    Returns:
        Dictionary of comparison results
    """
    results = {
        'match': True,
        'differences': [],
        'missing_files': [],
        'extra_files': []
    }
    
    # Check for phenotype files
    orig_files = sorted(list(Path(original_output).glob('*_phenotypes.csv')))
    new_files = sorted(list(Path(new_output).glob('*_phenotypes.csv')))
    
    # Check for missing files
    orig_file_names = {os.path.basename(f) for f in orig_files}
    new_file_names = {os.path.basename(f) for f in new_files}
    results['missing_files'] = list(orig_file_names - new_file_names)
    results['extra_files'] = list(new_file_names - orig_file_names)
    
    # Compare common files
    common_files = orig_file_names.intersection(new_file_names)
    for file_name in common_files:
        orig_path = Path(original_output) / file_name
        new_path = Path(new_output) / file_name
        
        orig_df = pd.read_csv(orig_path)
        new_df = pd.read_csv(new_path)
        
        # Check row count
        if len(orig_df) != len(new_df):
            results['match'] = False
            results['differences'].append(f"{file_name}: Row count mismatch - "
                                         f"original {len(orig_df)} vs new {len(new_df)}")
            continue
        
        # Compare cell IDs
        if 'cell_id' in orig_df.columns and 'cell_id' in new_df.columns:
            orig_cells = set(orig_df['cell_id'])
            new_cells = set(new_df['cell_id'])
            
            missing_cells = orig_cells - new_cells
            extra_cells = new_cells - orig_cells
            
            if missing_cells:
                results['match'] = False
                results['differences'].append(f"{file_name}: Missing cells - count: {len(missing_cells)}")
                
            if extra_cells:
                results['match'] = False
                results['differences'].append(f"{file_name}: Extra cells - count: {len(extra_cells)}")
        
        # Find common numerical columns for comparison
        orig_numeric_cols = orig_df.select_dtypes(include=['number']).columns
        new_numeric_cols = new_df.select_dtypes(include=['number']).columns
        common_numeric_cols = set(orig_numeric_cols).intersection(set(new_numeric_cols))
        
        # Sort by cell_id first to ensure alignment
        if 'cell_id' in orig_df.columns and 'cell_id' in new_df.columns:
            orig_df = orig_df.sort_values('cell_id').reset_index(drop=True)
            new_df = new_df.sort_values('cell_id').reset_index(drop=True)
        
        # Compare numerical columns
        column_differences = []
        for col in common_numeric_cols:
            # Skip cell_id
            if col == 'cell_id':
                continue
                
            # Skip columns with zeros to avoid division by zero
            if (orig_df[col] == 0).all() or (new_df[col] == 0).all():
                continue
            
            # Calculate absolute percentage difference
            orig_vals = orig_df[col].values
            new_vals = new_df[col].values
            
            # Handle potential zeros by adding a small epsilon
            epsilon = 1e-10
            rel_diff = np.abs((orig_vals - new_vals) / (np.abs(orig_vals) + epsilon))
            mean_rel_diff = np.mean(rel_diff) * 100  # Convert to percentage
            
            if mean_rel_diff > tolerance * 100:
                column_differences.append((col, mean_rel_diff))
        
        # Add column differences to results
        if column_differences:
            results['match'] = False
            diff_msg = f"{file_name}: Column differences - "
            for col, diff in column_differences:
                diff_msg += f"{col}: {diff:.2f}% avg diff, "
            results['differences'].append(diff_msg[:-2])  # Remove last comma and space
    
    return results

def run_test(args):
    """Run the phenotyping test.
    
    Args:
        args: Command-line arguments
    """
    # Create test output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create pipeline
    pipeline = StandardPhenotypingPipeline(
        input_file=args.input_file,
        segmentation_dir=args.segmentation_dir,
        channels=args.channels.split(','),
        genotyping_dir=args.genotyping_dir if args.genotyping_dir else None,
        output_dir=args.output_dir,
    )
    
    # Run pipeline with a single well
    pipeline.run(wells=[args.well])
    
    # Compare results if original results are available
    if args.original_output:
        results = compare_phenotyping_results(args.original_output, args.output_dir)
        
        print("\n=== Phenotyping Comparison Results ===")
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
        description="Test phenotyping pipeline against original implementation"
    )
    
    parser.add_argument(
        '--input-file',
        required=True,
        help="Path to input ND2 file"
    )
    
    parser.add_argument(
        '--segmentation-dir',
        required=True,
        help="Path to segmentation results"
    )
    
    parser.add_argument(
        '--channels',
        required=True,
        help="Comma-separated list of channel names (e.g. 'DAPI,GFP,RFP')"
    )
    
    parser.add_argument(
        '--genotyping-dir',
        help="Path to genotyping results (optional)"
    )
    
    parser.add_argument(
        '--well',
        default='Well1',
        help="Well to process (default: Well1)"
    )
    
    parser.add_argument(
        '--output-dir',
        default='./results/test_phenotyping',
        help="Output directory (default: ./results/test_phenotyping)"
    )
    
    parser.add_argument(
        '--original-output',
        help="Path to original implementation output for comparison"
    )
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run_test(args)