"""Command-line interface for the image analysis pipeline."""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import argparse

from core.segmentation.segmentation_10x import Segmentation10XPipeline
from core.segmentation.segmentation_40x import Segmentation40XPipeline
from core.genotyping.pipeline import StandardGenotypingPipeline
from core.phenotyping.pipeline import StandardPhenotypingPipeline
from core.visualization.albums import AlbumGenerationPipeline

def setup_logging(output_dir: str, level: int = logging.INFO) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        output_dir: Directory for log file
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('image_analysis')
    logger.setLevel(level)
    
    # Create handlers
    log_file = os.path.join(output_dir, 'pipeline.log')
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file.
    
    Args:
        config_file: Path to JSON configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid
    """
    with open(config_file, 'r') as f:
        return json.load(f)

def run_segmentation(
    args: argparse.Namespace,
    config: Dict[str, Any],
    logger: logging.Logger
) -> None:
    """Run the segmentation pipeline.
    
    Args:
        args: Command-line arguments
        config: Configuration dictionary
        logger: Logger instance
    """
    # Determine magnification and pipeline
    if args.magnification == '10x':
        pipeline_class = Segmentation10XPipeline
    elif args.magnification == '40x':
        pipeline_class = Segmentation40XPipeline
    else:
        raise ValueError(f"Invalid magnification: {args.magnification}")
        
    # Create pipeline
    pipeline = pipeline_class(
        input_file=args.input_file,
        output_dir=args.output_dir,
        config_file=args.config,
        **config.get('segmentation', {})
    )
    
    # Run pipeline
    pipeline.run(wells=args.wells)

def run_genotyping(
    args: argparse.Namespace,
    config: Dict[str, Any],
    logger: logging.Logger
) -> None:
    """Run the genotyping pipeline.
    
    Args:
        args: Command-line arguments
        config: Configuration dictionary
        logger: Logger instance
    """
    # Create pipeline
    pipeline = StandardGenotypingPipeline(
        input_file=args.input_file,
        segmentation_dir=args.segmentation_dir,
        barcode_library=args.barcode_library,
        output_dir=args.output_dir,
        config_file=args.config,
        **config.get('genotyping', {})
    )
    
    # Run pipeline
    pipeline.run(wells=args.wells)

def run_phenotyping(
    args: argparse.Namespace,
    config: Dict[str, Any],
    logger: logging.Logger
) -> None:
    """Run the phenotyping pipeline.
    
    Args:
        args: Command-line arguments
        config: Configuration dictionary
        logger: Logger instance
    """
    # Create pipeline
    pipeline = StandardPhenotypingPipeline(
        input_file=args.input_file,
        segmentation_dir=args.segmentation_dir,
        channels=args.channels,
        genotyping_dir=args.genotyping_dir,
        output_dir=args.output_dir,
        config_file=args.config,
        **config.get('phenotyping', {})
    )
    
    # Run pipeline
    pipeline.run(wells=args.wells)

def run_albums(
    args: argparse.Namespace,
    config: Dict[str, Any],
    logger: logging.Logger
) -> None:
    """Run the album generation pipeline.
    
    Args:
        args: Command-line arguments
        config: Configuration dictionary
        logger: Logger instance
    """
    # Create pipeline
    pipeline = AlbumGenerationPipeline(
        input_file=args.input_file,
        phenotyping_dir=args.phenotyping_dir,
        channels=args.channels,
        output_dir=args.output_dir,
        config_file=args.config,
        **config.get('albums', {})
    )
    
    # Run pipeline
    pipeline.run(wells=args.wells)

def run_full_pipeline(
    args: argparse.Namespace,
    config: Dict[str, Any],
    logger: logging.Logger
) -> None:
    """Run the complete analysis pipeline.
    
    Args:
        args: Command-line arguments
        config: Configuration dictionary
        logger: Logger instance
    """
    # Run segmentation
    logger.info("Starting segmentation pipeline...")
    run_segmentation(args, config, logger)
    
    # Run genotyping if requested
    if not args.skip_genotyping:
        logger.info("Starting genotyping pipeline...")
        run_genotyping(args, config, logger)
    
    # Run phenotyping
    logger.info("Starting phenotyping pipeline...")
    run_phenotyping(args, config, logger)
    
    # Generate albums if requested
    if not args.skip_albums:
        logger.info("Starting album generation...")
        run_albums(args, config, logger)
        
    logger.info("Full pipeline completed successfully")

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Image analysis pipeline for cellular phenotyping"
    )
    
    # Common arguments
    parser.add_argument(
        'input_file',
        help="Path to input ND2 file"
    )
    parser.add_argument(
        '--output-dir',
        help="Output directory (default: ./results)",
        default='./results'
    )
    parser.add_argument(
        '--config',
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        '--wells',
        nargs='+',
        help="List of wells to process (default: all)"
    )
    parser.add_argument(
        '--channels',
        nargs='+',
        help="List of channel names in order"
    )
    
    # Pipeline selection
    parser.add_argument(
        '--pipeline',
        choices=['full', 'segmentation', 'genotyping', 'phenotyping', 'albums'],
        default='full',
        help="Pipeline to run (default: full)"
    )
    
    # Segmentation arguments
    parser.add_argument(
        '--magnification',
        choices=['10x', '40x'],
        default='10x',
        help="Image magnification (default: 10x)"
    )
    
    # Genotyping arguments
    parser.add_argument(
        '--segmentation-dir',
        help="Directory containing segmentation results"
    )
    parser.add_argument(
        '--barcode-library',
        help="Path to barcode library CSV file"
    )
    
    # Phenotyping arguments
    parser.add_argument(
        '--genotyping-dir',
        help="Directory containing genotyping results"
    )
    
    # Album arguments
    parser.add_argument(
        '--phenotyping-dir',
        help="Directory containing phenotyping results"
    )
    
    # Full pipeline options
    parser.add_argument(
        '--skip-genotyping',
        action='store_true',
        help="Skip genotyping step in full pipeline"
    )
    parser.add_argument(
        '--skip-albums',
        action='store_true',
        help="Skip album generation in full pipeline"
    )
    
    return parser.parse_args()

def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments.
    
    Args:
        args: Parsed arguments
        
    Raises:
        ValueError: If arguments are invalid
    """
    # Check input file exists
    if not os.path.exists(args.input_file):
        raise ValueError(f"Input file not found: {args.input_file}")
        
    # Check config file if provided
    if args.config and not os.path.exists(args.config):
        raise ValueError(f"Config file not found: {args.config}")
        
    # Validate pipeline-specific arguments
    if args.pipeline in ['genotyping', 'phenotyping', 'full']:
        if not args.segmentation_dir:
            raise ValueError("Segmentation directory required")
            
    if args.pipeline in ['genotyping', 'full']:
        if not args.barcode_library:
            raise ValueError("Barcode library required")
            
    if args.pipeline in ['phenotyping', 'albums', 'full']:
        if not args.channels:
            raise ValueError("Channel names required")
            
    if args.pipeline == 'albums':
        if not args.phenotyping_dir:
            raise ValueError("Phenotyping directory required")

def main() -> None:
    """Main entry point for the CLI."""
    # Parse arguments
    args = parse_args()
    
    try:
        # Validate arguments
        validate_args(args)
        
        # Set up logging
        logger = setup_logging(args.output_dir)
        logger.info(f"Starting pipeline: {args.pipeline}")
        
        # Load configuration
        config = {}
        if args.config:
            config = load_config(args.config)
            
        # Run selected pipeline
        if args.pipeline == 'full':
            run_full_pipeline(args, config, logger)
        elif args.pipeline == 'segmentation':
            run_segmentation(args, config, logger)
        elif args.pipeline == 'genotyping':
            run_genotyping(args, config, logger)
        elif args.pipeline == 'phenotyping':
            run_phenotyping(args, config, logger)
        elif args.pipeline == 'albums':
            run_albums(args, config, logger)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)
        
if __name__ == '__main__':
    main() 