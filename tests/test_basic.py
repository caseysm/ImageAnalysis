#!/usr/bin/env python3
"""Basic test script for the ImageAnalysis pipeline."""

import os
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from imageanalysis.utils.io import ImageLoader
from imageanalysis.core.segmentation.segmentation_10x import Segmentation10XPipeline

def test_segmentation():
    """Test the 10x segmentation pipeline."""
    # Find a test image
    data_dir = Path(__file__).parent.parent / 'data'
    pheno_dir = data_dir / 'phenotyping'
    
    # Use the first image found
    test_images = list(pheno_dir.glob('*.nd2'))
    if not test_images:
        print("No test images found in data directory.")
        return False
        
    test_image = test_images[0]
    print(f"Using test image: {test_image}")
    
    # Set up output directory
    output_dir = Path(__file__).parent.parent / 'results' / 'test_segmentation'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create pipeline
    pipeline = Segmentation10XPipeline(
        input_file=str(test_image),
        output_dir=str(output_dir)
    )
    
    # Run pipeline
    try:
        pipeline.run(wells=['Well1'])
        print("Segmentation pipeline completed successfully.")
        return True
    except Exception as e:
        print(f"Error running segmentation pipeline: {e}")
        return False
        
def main():
    """Run all tests."""
    print("Running basic tests...")
    
    success = True
    
    # Test segmentation
    print("\n=== Testing Segmentation ===")
    if not test_segmentation():
        success = False
        
    # Print summary
    print("\n=== Test Summary ===")
    if success:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed.")
        return 1
        
if __name__ == "__main__":
    sys.exit(main())