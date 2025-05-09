#!/usr/bin/env python3
"""Script for automated coordinate mapping between magnifications."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from imageanalysis.core.mapping.pipeline import MappingPipeline
from imageanalysis.utils.logging import setup_logger

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run coordinate mapping between 10X and 40X images"
    )
    
    # Required arguments
    parser.add_argument(
        "--seg-10x-dir",
        required=True,
        help="Directory containing 10X segmentation results"
    )
    parser.add_argument(
        "--seg-40x-dir",
        required=True,
        help="Directory containing 40X segmentation results"
    )
    
    # Optional arguments
    parser.add_argument(
        "--wells",
        nargs="+",
        help="List of wells to process (default: all wells)"
    )
    parser.add_argument(
        "--output-dir",
        default="results/mapping",
        help="Output directory (default: results/mapping)"
    )
    parser.add_argument(
        "--config",
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--log-file",
        help="Path to log file (default: output_dir/mapping.log)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)

def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_file = args.log_file or os.path.join(args.output_dir, "mapping.log")
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger(__name__, log_file, log_level)
    
    # Load configuration if provided
    config = None
    if args.config:
        try:
            config = load_config(args.config)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)
    
    # Initialize pipeline
    try:
        pipeline = MappingPipeline(
            seg_10x_dir=args.seg_10x_dir,
            seg_40x_dir=args.seg_40x_dir,
            output_dir=args.output_dir,
            config=config,
            logger=logger
        )
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    # Run pipeline
    try:
        results = pipeline.run(wells=args.wells)
        logger.info(f"Successfully processed {len(results)} wells")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)
    
    # Print summary
    logger.info("\nMapping Results Summary:")
    for well, result in results.items():
        logger.info(f"\nWell {well}:")
        logger.info(f"  Matched points: {len(result.matched_points_10x)}")
        logger.info(f"  RMSE: {result.error_metrics['rmse']:.2f} pixels")
        logger.info(f"  Max error: {result.error_metrics['max_error']:.2f} pixels")
        logger.info(f"  Results saved to: {args.output_dir}/{well}_mapping.json")
        logger.info(f"  Diagnostics saved to: {args.output_dir}/{well}_diagnostics/")

if __name__ == "__main__":
    main() 