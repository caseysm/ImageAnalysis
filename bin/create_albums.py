#!/usr/bin/env python3
"""Command-line script for generating cell image albums."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from core.visualization.albums import AlbumGenerationPipeline

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
        '--genotyping-dir',
        type=str,
        required=True,
        help='Directory containing genotyping results'
    )
    
    # Optional arguments
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
            
    # Create pipeline
    pipeline = AlbumGenerationPipeline(
        input_file=args.input_file,
        segmentation_dir=args.segmentation_dir,
        genotyping_dir=args.genotyping_dir,
        output_dir=args.output_dir,
        channels=args.channels,
        window_size=args.window_size,
        min_cells_per_guide=args.min_cells_per_guide,
        max_cells_per_guide=args.max_cells_per_guide,
        min_quality_score=args.min_quality_score,
        min_intensity_threshold=args.min_intensity_threshold,
        max_intensity_threshold=args.max_intensity_threshold
    )
    
    # Validate inputs
    try:
        pipeline.validate_inputs()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Run pipeline
    try:
        pipeline.run(guides=args.guides)
    except Exception as e:
        print(f"Error during pipeline execution: {e}", file=sys.stderr)
        sys.exit(1)
        
    print("Album generation completed successfully!")
    
if __name__ == '__main__':
    main() 