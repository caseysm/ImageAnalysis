#!/usr/bin/env python3
"""Command-line script for generating cell image albums."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from imageanalysis.core.visualization.albums import AlbumGenerationPipeline

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Generate albums of cell images for each guide RNA.'
    )
    
    # Required arguments
    parser.add_argument(
        '--phenotyping-dir',
        type=str,
        required=True,
        help='Directory containing phenotyping results'
    )
    
    # Optional arguments
    parser.add_argument(
        '--wells',
        type=str,
        nargs='+',
        help='List of wells to process (defaults to all wells)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (defaults to segmentation_dir/albums)'
    )
    parser.add_argument(
        '--channels',
        type=str,
        nargs='+',
        help='List of channel names in order they appear in ND2 file'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=64,
        help='Size of cell image windows'
    )
    parser.add_argument(
        '--min-cells-per-guide',
        type=int,
        default=10,
        help='Minimum number of cells per guide for inclusion'
    )
    parser.add_argument(
        '--max-cells-per-guide',
        type=int,
        default=100,
        help='Maximum number of cells per guide to include'
    )
    parser.add_argument(
        '--min-quality-score',
        type=float,
        default=0.8,
        help='Minimum quality score for barcode calls'
    )
    parser.add_argument(
        '--min-intensity-threshold',
        type=float,
        default=0.2,
        help='Minimum intensity threshold for normalization'
    )
    parser.add_argument(
        '--max-intensity-threshold',
        type=float,
        default=0.98,
        help='Maximum intensity threshold for normalization'
    )
    parser.add_argument(
        '--guides',
        type=str,
        nargs='+',
        help='List of specific guides to process'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to JSON configuration file'
    )
    
    return parser.parse_args()

def main() -> None:
    """Run the album generation pipeline."""
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
        'phenotyping_dir': args.phenotyping_dir,
        'output_dir': args.output_dir,
        'channels': args.channels,
        'window_size': args.window_size if hasattr(args, 'window_size') else 64,
        'wells': args.wells
    }
    
    # Create pipeline
    pipeline = AlbumGenerationPipeline(pipeline_config)
    
    # Run pipeline
    try:
        result = pipeline.run()
        print(f"Album generation results: {result}")
    except Exception as e:
        print(f"Error during pipeline execution: {e}", file=sys.stderr)
        sys.exit(1)
        
    print("Album generation completed successfully!")
    
if __name__ == '__main__':
    main() 