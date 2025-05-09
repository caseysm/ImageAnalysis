#!/usr/bin/env python3
"""Test script for validating genotyping pipeline against original implementation."""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directories to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.genotyping.pipeline import StandardGenotypingPipeline

def compare_genotyping_results(original_output, new_output, tolerance=0.01):
    """Compare genotyping results between original and new implementations.
    
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
    
    # Check for genotype assignment files
    orig_files = sorted(list(Path(original_output).glob('*_genotypes.csv')))
    new_files = sorted(list(Path(new_output).glob('*_genotypes.csv')))
    
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
        
        # Compare genotype assignments
        if 'genotype' in orig_df.columns and 'genotype' in new_df.columns:
            # Sort by cell_id first to ensure alignment
            if 'cell_id' in orig_df.columns and 'cell_id' in new_df.columns:
                orig_df = orig_df.sort_values('cell_id').reset_index(drop=True)
                new_df = new_df.sort_values('cell_id').reset_index(drop=True)
            
            mismatch_count = (orig_df['genotype'] != new_df['genotype']).sum()
            mismatch_percent = mismatch_count / len(orig_df) * 100
            
            if mismatch_percent > tolerance * 100:
                results['match'] = False
                results['differences'].append(f"{file_name}: Genotype mismatch - "
                                             f"{mismatch_count} cells ({mismatch_percent:.2f}%)")
        
        # Compare quality scores if present
        if 'quality_score' in orig_df.columns and 'quality_score' in new_df.columns:
            orig_df = orig_df.sort_values('cell_id').reset_index(drop=True)
            new_df = new_df.sort_values('cell_id').reset_index(drop=True)
            
            # Calculate mean absolute difference in quality scores
            quality_diff = np.abs(orig_df['quality_score'] - new_df['quality_score']).mean()
            
            if quality_diff > tolerance:
                results['match'] = False
                results['differences'].append(f"{file_name}: Quality score difference - "
                                             f"mean abs diff: {quality_diff:.4f}")
    
    return results

def run_test(args):
    """Run the genotyping test.
    
    Args:
        args: Command-line arguments
    """
    # Create test output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create pipeline
    pipeline = StandardGenotypingPipeline(
        input_file=args.input_file,
        segmentation_dir=args.segmentation_dir,
        barcode_library=args.barcode_library,
        output_dir=args.output_dir,
    )
    
    # Run pipeline with a single well
    pipeline.run(wells=[args.well])
    
    # Compare results if original results are available
    if args.original_output:
        results = compare_genotyping_results(args.original_output, args.output_dir)
        
        print("\n=== Genotyping Comparison Results ===")
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
        description="Test genotyping pipeline against original implementation"
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
        '--barcode-library',
        required=True,
        help="Path to barcode library CSV file"
    )
    
    parser.add_argument(
        '--well',
        default='Well1',
        help="Well to process (default: Well1)"
    )
    
    parser.add_argument(
        '--output-dir',
        default='./results/test_genotyping',
        help="Output directory (default: ./results/test_genotyping)"
    )
    
    parser.add_argument(
        '--original-output',
        help="Path to original implementation output for comparison"
    )
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run_test(args)