#!/usr/bin/env python3
"""
Test script for the refactored pipeline.
This runs the refactored implementation on real data.
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('refactored_test')

def test_segmentation(input_file, output_dir, is_10x=True):
    """Test the segmentation pipeline."""
    logger.info(f"Testing segmentation with file: {input_file}")
    
    try:
        # Import the appropriate pipeline
        if is_10x:
            from imageanalysis.core.segmentation import Segmentation10XPipeline as SegmentationPipeline
        else:
            from imageanalysis.core.segmentation import Segmentation40XPipeline as SegmentationPipeline
        
        # Create config
        config = {
            'input_file': str(input_file),
            'output_dir': output_dir,
            'nuclear_channel': 0,  # DAPI
            'cell_channel': 1      # Cell body
        }
        
        # Create and run pipeline
        pipeline = SegmentationPipeline(config)
        result = pipeline.run()
        
        logger.info(f"Segmentation result: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Error running segmentation: {str(e)}")
        return None

def test_genotyping(input_file, segmentation_dir, barcode_library, output_dir):
    """Test the genotyping pipeline."""
    logger.info(f"Testing genotyping with file: {input_file}")
    
    try:
        # Import the genotyping pipeline
        from imageanalysis.core.genotyping.pipeline import StandardGenotypingPipeline
        
        # Create config
        config = {
            'input_file': str(input_file),
            'segmentation_dir': segmentation_dir,
            'barcode_library': str(barcode_library),
            'output_dir': output_dir,
            'peak_threshold': 200,
            'min_quality_score': 0.3,
            'max_hamming_distance': 1
        }
        
        # Create and run pipeline
        pipeline = StandardGenotypingPipeline(config)
        result = pipeline.run()
        
        logger.info(f"Genotyping result: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Error running genotyping: {str(e)}")
        return None

def test_phenotyping(input_file, segmentation_dir, genotyping_dir, output_dir, channels):
    """Test the phenotyping pipeline."""
    logger.info(f"Testing phenotyping with file: {input_file}")
    
    try:
        # Import the phenotyping pipeline
        from imageanalysis.core.phenotyping.pipeline import PhenotypingPipeline
        
        # Create config
        config = {
            'input_file': str(input_file),
            'segmentation_dir': segmentation_dir,
            'genotyping_dir': genotyping_dir,
            'output_dir': output_dir,
            'channels': channels
        }
        
        # Create and run pipeline
        pipeline = PhenotypingPipeline(config)
        result = pipeline.run()
        
        logger.info(f"Phenotyping result: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Error running phenotyping: {str(e)}")
        return None

def main():
    """Main function."""
    # Define paths
    data_dir = Path("/Users/csm70/Desktop/Shalem_Lab/ImageAnalysis/Data")
    output_dir = Path("/Users/csm70/Desktop/Shalem_Lab/ImageAnalysis/results/refactored_test")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find files
    phenotyping_files = sorted(list(data_dir.glob('phenotyping/*.nd2')))
    genotyping_files = sorted(list(data_dir.glob('genotyping/cycle_1/*.nd2')))
    barcode_library = list(data_dir.glob('*.csv'))[0] if list(data_dir.glob('*.csv')) else None
    
    # Limit to first file for testing
    if phenotyping_files:
        pheno_file = phenotyping_files[0]
    else:
        logger.error("No phenotyping files found")
        return 1
    
    if genotyping_files:
        geno_file = genotyping_files[0]
    else:
        logger.error("No genotyping files found")
        return 1
    
    # Create specific output directories
    seg_dir = output_dir / 'segmentation'
    geno_dir = output_dir / 'genotyping'
    pheno_dir = output_dir / 'phenotyping'
    
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(geno_dir, exist_ok=True)
    os.makedirs(pheno_dir, exist_ok=True)
    
    # Run segmentation
    logger.info("=== Running Segmentation ===")
    seg_result = test_segmentation(pheno_file, seg_dir)
    
    # Run genotyping
    if seg_result and barcode_library:
        logger.info("=== Running Genotyping ===")
        geno_result = test_genotyping(geno_file, seg_dir, barcode_library, geno_dir)
    else:
        geno_result = None
    
    # Run phenotyping
    if seg_result:
        logger.info("=== Running Phenotyping ===")
        pheno_result = test_phenotyping(
            pheno_file, 
            seg_dir, 
            geno_dir if geno_result else None, 
            pheno_dir,
            ['DAPI', 'mClov3', 'TMR']
        )
    else:
        pheno_result = None
    
    # Print summary
    logger.info("\n=== Test Summary ===")
    logger.info(f"Segmentation: {'SUCCESS' if seg_result else 'FAILED'}")
    logger.info(f"Genotyping: {'SUCCESS' if geno_result else 'FAILED'}")
    logger.info(f"Phenotyping: {'SUCCESS' if pheno_result else 'FAILED'}")
    
    if seg_result and (geno_result or barcode_library is None) and pheno_result:
        logger.info("All tests passed successfully!")
        return 0
    else:
        logger.info("Some tests failed. Check logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())