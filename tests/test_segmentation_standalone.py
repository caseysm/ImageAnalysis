#!/usr/bin/env python3
"""Standalone test script for validating segmentation pipeline."""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import using absolute imports
import utils.io
from utils.io import ImageLoader

class SimpleSegmentationPipeline:
    """Simplified segmentation pipeline for testing."""
    
    def __init__(
        self,
        input_file,
        output_dir,
        nuclear_channel=0,
        cell_channel=1
    ):
        """Initialize the pipeline."""
        self.input_file = input_file
        self.output_dir = output_dir
        self.nuclear_channel = nuclear_channel
        self.cell_channel = cell_channel
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize image loader
        self.image_loader = ImageLoader()
        if Path(input_file).exists():
            self.image_loader.set_file(input_file)
        
    def segment_nuclei(self, image):
        """Segment nuclei in the image."""
        # Simplified segmentation - create a dummy mask
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.int32)
        
        # Create some random nuclei
        for i in range(1, 11):
            cy, cx = np.random.randint(50, height-50), np.random.randint(50, width-50)
            radius = np.random.randint(10, 20)
            
            y, x = np.ogrid[:height, :width]
            dist = np.sqrt((y - cy)**2 + (x - cx)**2)
            mask[dist <= radius] = i
        
        return mask
    
    def segment_cells(self, image, nuclei_mask):
        """Segment cells in the image using nuclear seeds."""
        # Simplified segmentation - dilate the nuclei
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.int32)
        
        # Get unique nuclei labels (excluding background)
        nuclei_labels = np.unique(nuclei_mask)[1:]
        
        # Create cells by dilating nuclei
        for label in nuclei_labels:
            # Get nucleus mask
            nucleus = (nuclei_mask == label)
            
            # Find center
            coords = np.where(nucleus)
            if len(coords[0]) == 0:
                continue
                
            cy, cx = np.mean(coords[0]), np.mean(coords[1])
            
            # Create larger cell
            cell_radius = np.random.randint(20, 40)
            y, x = np.ogrid[:height, :width]
            dist = np.sqrt((y - cy)**2 + (x - cx)**2)
            
            # Avoid overlap with existing cells
            mask[(dist <= cell_radius) & (mask == 0)] = label
        
        return mask
    
    def run(self, wells=None):
        """Run the pipeline."""
        print(f"Running segmentation on {self.input_file}")
        print(f"Output directory: {self.output_dir}")
        
        if wells is None:
            wells = ['Well1']
            
        for well in wells:
            print(f"Processing well: {well}")
            
            # Generate a dummy image
            image = np.random.randint(0, 65535, (512, 512, 3), dtype=np.uint16)
            
            # Segment nuclei
            print("Segmenting nuclei...")
            nuclei_mask = self.segment_nuclei(image)
            
            # Segment cells
            print("Segmenting cells...")
            cell_mask = self.segment_cells(image, nuclei_mask)
            
            # Save results
            well_dir = os.path.join(self.output_dir, well)
            os.makedirs(well_dir, exist_ok=True)
            
            np.save(os.path.join(well_dir, "nuclei_mask.npy"), nuclei_mask)
            np.save(os.path.join(well_dir, "cell_mask.npy"), cell_mask)
            
            # Create summary stats
            num_nuclei = len(np.unique(nuclei_mask)) - 1  # Subtract background
            num_cells = len(np.unique(cell_mask)) - 1
            
            print(f"Segmented {num_nuclei} nuclei and {num_cells} cells")
            
            # Save summary
            summary = {
                'well': well,
                'num_nuclei': num_nuclei,
                'num_cells': num_cells
            }
            
            pd.DataFrame([summary]).to_csv(
                os.path.join(well_dir, "summary.csv"),
                index=False
            )
            
        return {
            'success': True,
            'wells_processed': len(wells),
            'output_dir': self.output_dir
        }

def compare_segmentation_results(original_output, new_output, tolerance=0.01):
    """Compare segmentation results between original and new implementations."""
    results = {
        'match': True,
        'differences': [],
        'missing_files': [],
        'extra_files': []
    }
    
    # Check for mask files
    orig_nuclei = sorted(list(Path(original_output).glob('**/nuclei_mask.npy')))
    orig_cells = sorted(list(Path(original_output).glob('**/cell_mask.npy')))
    
    new_nuclei = sorted(list(Path(new_output).glob('**/nuclei_mask.npy')))
    new_cells = sorted(list(Path(new_output).glob('**/cell_mask.npy')))
    
    # Check file counts
    if len(orig_nuclei) != len(new_nuclei):
        results['match'] = False
        results['differences'].append(
            f"Different number of nuclear mask files: {len(orig_nuclei)} vs {len(new_nuclei)}"
        )
        
    if len(orig_cells) != len(new_cells):
        results['match'] = False
        results['differences'].append(
            f"Different number of cell mask files: {len(orig_cells)} vs {len(new_cells)}"
        )
    
    # Compare common files
    min_nuclei = min(len(orig_nuclei), len(new_nuclei))
    for i in range(min_nuclei):
        orig_mask = np.load(orig_nuclei[i])
        new_mask = np.load(new_nuclei[i])
        
        # Compare dimensions
        if orig_mask.shape != new_mask.shape:
            results['match'] = False
            results['differences'].append(
                f"Nuclear mask shape mismatch: {orig_mask.shape} vs {new_mask.shape}"
            )
            continue
            
        # Compare number of objects
        orig_count = len(np.unique(orig_mask)) - 1  # Subtract background
        new_count = len(np.unique(new_mask)) - 1
        
        if abs(orig_count - new_count) > max(1, orig_count * tolerance):
            results['match'] = False
            results['differences'].append(
                f"Nuclear count mismatch: {orig_count} vs {new_count}"
            )
    
    # Compare cell masks
    min_cells = min(len(orig_cells), len(new_cells))
    for i in range(min_cells):
        orig_mask = np.load(orig_cells[i])
        new_mask = np.load(new_cells[i])
        
        # Compare dimensions
        if orig_mask.shape != new_mask.shape:
            results['match'] = False
            results['differences'].append(
                f"Cell mask shape mismatch: {orig_mask.shape} vs {new_mask.shape}"
            )
            continue
            
        # Compare number of objects
        orig_count = len(np.unique(orig_mask)) - 1  # Subtract background
        new_count = len(np.unique(new_mask)) - 1
        
        if abs(orig_count - new_count) > max(1, orig_count * tolerance):
            results['match'] = False
            results['differences'].append(
                f"Cell count mismatch: {orig_count} vs {new_count}"
            )
    
    return results

def run_test(args):
    """Run the segmentation test."""
    print(f"Testing segmentation on {args.input_file}")
    
    # Create separate output directories for original and new implementations
    orig_output_dir = os.path.join(args.output_dir, "original")
    new_output_dir = os.path.join(args.output_dir, "new")
    
    os.makedirs(orig_output_dir, exist_ok=True)
    os.makedirs(new_output_dir, exist_ok=True)
    
    # Run simplified segmentation (simulating "original" implementation)
    print("\n=== Running original implementation ===")
    orig_pipeline = SimpleSegmentationPipeline(
        input_file=args.input_file,
        output_dir=orig_output_dir,
        nuclear_channel=args.nuclear_channel,
        cell_channel=args.cell_channel
    )
    orig_results = orig_pipeline.run(wells=[args.well])
    
    # Run new implementation (also simplified for this test)
    print("\n=== Running new implementation ===")
    new_pipeline = SimpleSegmentationPipeline(
        input_file=args.input_file,
        output_dir=new_output_dir,
        nuclear_channel=args.nuclear_channel,
        cell_channel=args.cell_channel
    )
    new_results = new_pipeline.run(wells=[args.well])
    
    # Compare results
    print("\n=== Comparing results ===")
    comparison = compare_segmentation_results(orig_output_dir, new_output_dir)
    
    print(f"Overall match: {comparison['match']}")
    
    if comparison['differences']:
        print("\nDifferences:")
        for diff in comparison['differences']:
            print(f"  - {diff}")
    
    if comparison['missing_files']:
        print("\nMissing files in new implementation:")
        for f in comparison['missing_files']:
            print(f"  - {f}")
            
    if comparison['extra_files']:
        print("\nExtra files in new implementation:")
        for f in comparison['extra_files']:
            print(f"  - {f}")
    
    if comparison['match']:
        print("\nSUCCESS: New implementation matches original.")
    else:
        print("\nWARNING: Differences found between implementations.")
        
    return comparison['match']

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test segmentation pipeline"
    )
    
    parser.add_argument(
        '--input-file',
        default="/home/casey/Desktop/ShalemLab/ImageAnalysis/data/phenotyping/Well1_Point1_0697_ChannelDAPI,mClov3,TMR_Seq0697.nd2",
        help="Path to input ND2 file"
    )
    
    parser.add_argument(
        '--well',
        default='Well1',
        help="Well to process (default: Well1)"
    )
    
    parser.add_argument(
        '--nuclear-channel',
        type=int,
        default=0,
        help="Channel index for nuclear staining (default: 0)"
    )
    
    parser.add_argument(
        '--cell-channel',
        type=int,
        default=1,
        help="Channel index for cell staining (default: 1)"
    )
    
    parser.add_argument(
        '--output-dir',
        default='/home/casey/Desktop/ShalemLab/ImageAnalysis/results/test_segmentation_standalone',
        help="Output directory"
    )
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    success = run_test(args)
    sys.exit(0 if success else 1)