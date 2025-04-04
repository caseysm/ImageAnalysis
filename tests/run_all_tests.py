#!/usr/bin/env python3
"""Script to run all validation tests for the Image Analysis pipeline."""

import os
import sys
import argparse
from pathlib import Path
import subprocess

def parse_args():
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run validation tests for Image Analysis pipeline"
    )
    
    parser.add_argument(
        '--test',
        choices=['all', 'segmentation', 'genotyping', 'phenotyping'],
        default='all',
        help="Test to run (default: all)"
    )
    
    parser.add_argument(
        '--data-dir',
        default='./data',
        help="Data directory (default: ./data)"
    )
    
    parser.add_argument(
        '--output-dir',
        default='./results/tests',
        help="Output directory (default: ./results/tests)"
    )
    
    parser.add_argument(
        '--original-output',
        help="Path to original implementation output for comparison"
    )
    
    return parser.parse_args()

def run_segmentation_test(args):
    """Run segmentation test.
    
    Args:
        args: Command-line arguments
    """
    print("\n=== Running Segmentation Test ===")
    
    # Find sample phenotyping image
    data_dir = Path(args.data_dir)
    pheno_images = list(data_dir.glob('phenotyping/*.nd2'))
    
    if not pheno_images:
        print("Error: No phenotyping images found in data directory.")
        return False
    
    test_image = str(pheno_images[0])
    print(f"Using test image: {test_image}")
    
    # Set up output directory
    output_dir = Path(args.output_dir) / 'segmentation'
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable,
        str(Path(__file__).parent / 'test_segmentation/test_segmentation.py'),
        f'--input-file={test_image}',
        f'--output-dir={output_dir}'
    ]
    
    if args.original_output:
        orig_seg_dir = Path(args.original_output) / 'segmentation'
        if orig_seg_dir.exists():
            cmd.append(f'--original-output={orig_seg_dir}')
    
    # Run test
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output
    print(result.stdout)
    if result.stderr:
        print("ERRORS:")
        print(result.stderr)
    
    return result.returncode == 0

def run_genotyping_test(args):
    """Run genotyping test.
    
    Args:
        args: Command-line arguments
    """
    print("\n=== Running Genotyping Test ===")
    
    # Find sample genotyping images
    data_dir = Path(args.data_dir)
    geno_dirs = list(data_dir.glob('genotyping/cycle_*'))
    
    if not geno_dirs:
        print("Error: No genotyping cycle directories found in data directory.")
        return False
    
    # Use first cycle
    cycle_dir = geno_dirs[0]
    geno_images = list(cycle_dir.glob('*.nd2'))
    
    if not geno_images:
        print(f"Error: No genotyping images found in {cycle_dir}.")
        return False
    
    test_image = str(geno_images[0])
    print(f"Using test image: {test_image}")
    
    # Find barcode library
    barcode_files = list(data_dir.glob('*.csv'))
    if not barcode_files:
        print("Error: No barcode library CSV found in data directory.")
        return False
    
    barcode_file = str(barcode_files[0])
    
    # Set up output directory
    output_dir = Path(args.output_dir) / 'genotyping'
    seg_dir = Path(args.output_dir) / 'segmentation'
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable,
        str(Path(__file__).parent / 'test_genotyping/test_genotyping.py'),
        f'--input-file={test_image}',
        f'--segmentation-dir={seg_dir}',
        f'--barcode-library={barcode_file}',
        f'--output-dir={output_dir}'
    ]
    
    if args.original_output:
        orig_geno_dir = Path(args.original_output) / 'genotyping'
        if orig_geno_dir.exists():
            cmd.append(f'--original-output={orig_geno_dir}')
    
    # Run test
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output
    print(result.stdout)
    if result.stderr:
        print("ERRORS:")
        print(result.stderr)
    
    return result.returncode == 0

def run_phenotyping_test(args):
    """Run phenotyping test.
    
    Args:
        args: Command-line arguments
    """
    print("\n=== Running Phenotyping Test ===")
    
    # Find sample phenotyping image
    data_dir = Path(args.data_dir)
    pheno_images = list(data_dir.glob('phenotyping/*.nd2'))
    
    if not pheno_images:
        print("Error: No phenotyping images found in data directory.")
        return False
    
    test_image = str(pheno_images[0])
    print(f"Using test image: {test_image}")
    
    # Set up output directory
    output_dir = Path(args.output_dir) / 'phenotyping'
    seg_dir = Path(args.output_dir) / 'segmentation'
    geno_dir = Path(args.output_dir) / 'genotyping'
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable,
        str(Path(__file__).parent / 'test_phenotyping/test_phenotyping.py'),
        f'--input-file={test_image}',
        f'--segmentation-dir={seg_dir}',
        f'--channels=DAPI,mClov3,TMR',
        f'--output-dir={output_dir}'
    ]
    
    if geno_dir.exists():
        cmd.append(f'--genotyping-dir={geno_dir}')
    
    if args.original_output:
        orig_pheno_dir = Path(args.original_output) / 'phenotyping'
        if orig_pheno_dir.exists():
            cmd.append(f'--original-output={orig_pheno_dir}')
    
    # Run test
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output
    print(result.stdout)
    if result.stderr:
        print("ERRORS:")
        print(result.stderr)
    
    return result.returncode == 0

def run_real_data_test(args):
    """Run a full pipeline test with real data.
    
    Args:
        args: Command-line arguments
        
    Returns:
        bool: True if the test passed, False otherwise
    """
    print("\n=== Running Real Data Pipeline Test ===")
    
    # Set up output directory
    output_dir = os.path.join(args.output_dir, "real_data_test")
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable,
        str(Path(__file__).parent / 'test_real_data_pipeline.py'),
        f'--output-dir={output_dir}',
        '--limit-files=2'  # Use just a few files to keep test fast
    ]
    
    # Run test
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output
    print(result.stdout)
    if result.stderr:
        print("ERRORS:")
        print(result.stderr)
    
    return result.returncode == 0

def main():
    """Main function."""
    args = parse_args()
    
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run tests
    success = True
    
    if args.test in ['all', 'segmentation']:
        if not run_segmentation_test(args):
            success = False
    
    if args.test in ['all', 'genotyping']:
        if not run_genotyping_test(args):
            success = False
    
    if args.test in ['all', 'phenotyping']:
        if not run_phenotyping_test(args):
            success = False
    
    # Always run the real data test
    if not run_real_data_test(args):
        success = False
    
    print("\n=== Test Summary ===")
    if success:
        print("All tests completed successfully.")
        return 0
    else:
        print("Some tests failed. Check output for details.")
        return 1

if __name__ == '__main__':
    sys.exit(main())