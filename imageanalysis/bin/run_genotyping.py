#!/usr/bin/env python3
"""Command-line script for running genotyping pipeline."""

import argparse
import json
import os
import sys
from pathlib import Path

from imageanalysis.core.genotyping.pipeline import StandardGenotypingPipeline
from imageanalysis.utils.logging import setup_logger

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run genotyping pipeline")
    
    parser.add_argument(
        "input_file",
        help="Path to input ND2 file"
    )
    
    parser.add_argument(
        "--segmentation-dir",
        required=True,
        help="Directory containing segmentation results"
    )
    
    parser.add_argument(
        "--barcode-library",
        required=True,
        help="Path to barcode library CSV file"
    )
    
    parser.add_argument(
        "--wells",
        nargs="+",
        help="List of wells to process (default: all wells)"
    )
    
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.8,
        help="Minimum quality score for base calls (default: 0.8)"
    )
    
    parser.add_argument(
        "--max-hamming-distance",
        type=int,
        default=1,
        help="Maximum Hamming distance for barcode matching (default: 1)"
    )
    
    parser.add_argument(
        "--min-peak-height",
        type=float,
        default=0.2,
        help="Minimum normalized peak height (default: 0.2)"
    )
    
    parser.add_argument(
        "--min-peak-distance",
        type=int,
        default=3,
        help="Minimum distance between peaks (default: 3)"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: results/genotyping)"
    )
    
    parser.add_argument(
        "--config",
        help="Path to JSON configuration file"
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    logger = setup_logger("genotyping")
    
    # Load configuration
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config = json.load(f)
            
    # Update config with command-line arguments
    config.update({
        'input_file': args.input_file,
        'segmentation_dir': args.segmentation_dir,
        'barcode_library': args.barcode_library,
        'min_quality_score': args.min_quality_score,
        'max_hamming_distance': args.max_hamming_distance,
        'min_peak_height': args.min_peak_height,
        'min_peak_distance': args.min_peak_distance,
        'output_dir': args.output_dir
    })
    
    if args.wells:
        config['wells'] = args.wells
        
    try:
        # Initialize and run pipeline
        pipeline = StandardGenotypingPipeline(config)
        results = pipeline.run()
        
        # Results are automatically saved by the pipeline
        logger.info("Genotyping pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 