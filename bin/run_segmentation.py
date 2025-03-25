#!/usr/bin/env python3
"""Command-line script for running image segmentation pipelines."""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to Python path to import local modules
sys.path.append(str(Path(__file__).parent.parent))

from core.segmentation.segmentation_10x import Segmentation10XPipeline
from core.segmentation.segmentation_40x import Segmentation40XPipeline
from utils.logging import setup_logger

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run image segmentation pipeline")
    
    parser.add_argument(
        "input_file",
        help="Path to input ND2 file"
    )
    
    parser.add_argument(
        "--magnification",
        choices=["10x", "40x"],
        required=True,
        help="Microscope magnification"
    )
    
    parser.add_argument(
        "--wells",
        nargs="+",
        help="List of wells to process (default: all wells)"
    )
    
    parser.add_argument(
        "--nuclear-channel",
        type=int,
        default=0,
        help="Channel index for nuclear staining (default: 0)"
    )
    
    parser.add_argument(
        "--cell-channel",
        type=int,
        default=1,
        help="Channel index for cell staining (default: 1)"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: results/segmentation)"
    )
    
    parser.add_argument(
        "--config",
        help="Path to JSON configuration file"
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    logger = setup_logger("segmentation")
    
    # Load configuration
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config = json.load(f)
            
    # Update config with command-line arguments
    config.update({
        'input_file': args.input_file,
        'nuclear_channel': args.nuclear_channel,
        'cell_channel': args.cell_channel,
        'output_dir': args.output_dir
    })
    
    if args.wells:
        config['wells'] = args.wells
        
    # Initialize appropriate pipeline
    if args.magnification == "10x":
        pipeline = Segmentation10XPipeline(config)
    else:
        pipeline = Segmentation40XPipeline(config)
        
    try:
        # Run pipeline
        results = pipeline.run()
        
        # Save results summary
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = os.path.join("results", "segmentation")
            
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, "segmentation_summary.json")
        
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Results saved to {summary_path}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 