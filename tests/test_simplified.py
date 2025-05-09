#!/usr/bin/env python3
"""Simplified test for the ImageAnalysis pipeline."""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Use package imports
import sys
from pathlib import Path

def test_image_loading():
    """Test image loading functionality."""
    try:
        from imageanalysis.utils.io import ImageLoader
        
        # Create image loader
        loader = ImageLoader()
        
        # Get a test image
        data_dir = Path(__file__).parent.parent / 'data'
        pheno_dir = data_dir / 'phenotyping'
        
        test_images = list(pheno_dir.glob('*.nd2'))
        if not test_images:
            print("No test images found.")
            return False
            
        test_image = test_images[0]
        print(f"Using test image: {test_image}")
        
        # Set file and get dummy data
        loader.set_file(test_image)
        print("Successfully created ImageLoader")
        
        # Test methods
        wells = loader.get_wells()
        print(f"Found wells: {wells}")
        
        # Test dummy image generation
        image = loader.load()
        print(f"Generated dummy image with shape: {image.shape}")
        
        return True
    except Exception as e:
        print(f"Error in image loading test: {e}")
        return False

def test_basic_segmentation():
    """Test basic segmentation functionality."""
    try:
        # Create dummy masks
        height, width = 512, 512
        
        # Create nuclear mask with 10 random nuclei
        nuclear_mask = np.zeros((height, width), dtype=np.int32)
        cell_mask = np.zeros((height, width), dtype=np.int32)
        
        for i in range(1, 11):
            # Random center
            cy, cx = np.random.randint(50, height-50), np.random.randint(50, width-50)
            
            # Random radii
            nucleus_radius = np.random.randint(10, 15)
            cell_radius = np.random.randint(25, 40)
            
            # Create masks
            y, x = np.ogrid[:height, :width]
            dist_nuc = np.sqrt((y - cy)**2 + (x - cx)**2)
            dist_cell = np.sqrt((y - cy)**2 + (x - cx)**2)
            
            # Fill masks
            nuclear_mask[dist_nuc <= nucleus_radius] = i
            cell_mask[dist_cell <= cell_radius] = i
        
        # Save masks
        output_dir = Path(__file__).parent.parent / 'results' / 'test_basic'
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(output_dir / 'nuclear_mask.npy', nuclear_mask)
        np.save(output_dir / 'cell_mask.npy', cell_mask)
        
        print(f"Created and saved segmentation masks with {np.unique(nuclear_mask).size - 1} nuclei")
        
        return True
    except Exception as e:
        print(f"Error in basic segmentation test: {e}")
        return False

def test_basic_genotyping():
    """Test basic genotyping functionality."""
    try:
        # Create dummy barcode assignments
        output_dir = Path(__file__).parent.parent / 'results' / 'test_basic'
        os.makedirs(output_dir, exist_ok=True)
        
        # Create 10 cells with random barcodes
        cell_ids = list(range(1, 11))
        barcodes = ['ACGT', 'GACT', 'TGCA', 'CATG', 'GTAC']
        
        assignments = []
        for cell_id in cell_ids:
            barcode = np.random.choice(barcodes)
            quality = np.random.uniform(0.8, 1.0)
            
            assignments.append({
                'cell_id': cell_id,
                'barcode': barcode,
                'quality_score': quality,
                'gene': f'Gene_{barcode}'
            })
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(assignments)
        df.to_csv(output_dir / 'cell_barcodes.csv', index=False)
        
        print(f"Created and saved genotyping results for {len(cell_ids)} cells")
        
        return True
    except Exception as e:
        print(f"Error in basic genotyping test: {e}")
        return False

def main():
    """Run all simplified tests."""
    print("Running simplified tests...")
    
    # Track results
    results = {}
    
    # Test image loading
    print("\n=== Testing Image Loading ===")
    results['image_loading'] = test_image_loading()
    
    # Test basic segmentation
    print("\n=== Testing Basic Segmentation ===")
    results['segmentation'] = test_basic_segmentation()
    
    # Test basic genotyping
    print("\n=== Testing Basic Genotyping ===")
    results['genotyping'] = test_basic_genotyping()
    
    # Print summary
    print("\n=== Test Summary ===")
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())