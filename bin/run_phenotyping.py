#!/usr/bin/env python3
"""Command-line script for running phenotype analysis."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from imageanalysis.core.phenotyping.pipeline import StandardPhenotypingPipeline

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Run phenotype analysis on segmented images.'
    )
    
    # Required arguments
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input ND2 file'
    )
    parser.add_argument(
        '--segmentation-dir',
        type=str,
        required=True,
        help='Directory containing segmentation results'
    )
    parser.add_argument(
        '--channels',
        type=str,
        nargs='+',
        required=True,
        help='List of channel names in order they appear in ND2 file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default=['area', 'intensity', 'shape', 'texture', 'location'],
        choices=['area', 'intensity', 'shape', 'texture', 'location'],
        help='List of metrics to calculate'
    )
    parser.add_argument(
        '--genotyping-dir',
        type=str,
        help='Directory containing genotyping results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (defaults to segmentation_dir/phenotyping)'
    )
    parser.add_argument(
        '--pixel-size',
        type=float,
        default=1.0,
        help='Physical size of each pixel in microns'
    )
    parser.add_argument(
        '--wells',
        type=str,
        nargs='+',
        help='List of wells to process (defaults to all wells)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to JSON configuration file'
    )
    
    return parser.parse_args()

def main() -> None:
    """Run the phenotyping pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration if provided
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
            
    # Update arguments with config values
    for key, value in config.items():
        if not getattr(args, key):
            setattr(args, key, value)
            
    # Create config dictionary for pipeline
    pipeline_config = {
        'input_file': args.input_file,
        'segmentation_dir': args.segmentation_dir,
        'channels': args.channels,
        'metrics': args.metrics,
        'genotyping_dir': args.genotyping_dir,
        'output_dir': args.output_dir,
        'pixel_size': args.pixel_size,
        'wells': args.wells
    }
    
    # Create pipeline
    pipeline = StandardPhenotypingPipeline(pipeline_config)
    
    # Run pipeline
    try:
        result = pipeline.run()
        print(f"Phenotyping results: {result}")
    except Exception as e:
        print(f"Error during pipeline execution: {e}", file=sys.stderr)
        sys.exit(1)
        
    print("Phenotype analysis completed successfully!")
    
if __name__ == '__main__':
    main() 