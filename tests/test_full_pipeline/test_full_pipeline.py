#!/usr/bin/env python3
"""Test script for the full ImageAnalysis pipeline on complete production data."""

import argparse
import os
import sys
import time
import logging
from pathlib import Path
import tempfile
import json

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test the full ImageAnalysis pipeline on production data"
    )
    
    parser.add_argument(
        "--data-dir",
        default="/home/casey/Desktop/ShalemLab/ImageAnalysis/results/production",
        help="Base directory containing production data"
    )
    
    parser.add_argument(
        "--output-dir",
        default="/home/casey/Desktop/ShalemLab/ImageAnalysis/results/full_pipeline_test",
        help="Output directory for test results"
    )
    
    parser.add_argument(
        "--wells",
        nargs="+",
        default=["Well1"],
        help="List of wells to process"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()

def setup_logging(output_dir, debug=False):
    """Set up logging for the test script.
    
    Args:
        output_dir: Directory to save log file
        debug: Whether to enable debug logging
    
    Returns:
        Logger instance
    """
    # Create log directory
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logger
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"full_pipeline_test_{timestamp}.log")
    
    logger = logging.getLogger("full_pipeline_test")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def run_segmentation(input_files, output_dir, wells, logger):
    """Run segmentation step of the pipeline.
    
    Args:
        input_files: List of input ND2 files
        output_dir: Output directory
        wells: List of wells to process
        logger: Logger instance
        
    Returns:
        bool: True if successful, False otherwise
    """
    from imageanalysis.core.segmentation.segmentation_10x import Segmentation10x
    from imageanalysis.core.segmentation.segmentation_40x import Segmentation40x
    
    logger.info("Running segmentation...")
    segmentation_dir = os.path.join(output_dir, "segmentation")
    os.makedirs(segmentation_dir, exist_ok=True)
    
    success = True
    
    # Process each input file
    for input_file in input_files:
        try:
            # Determine magnification based on filename or metadata
            # This is a simplistic approach - in a real scenario, use metadata
            if "10x" in input_file.lower():
                logger.info(f"Processing 10X file: {input_file}")
                segmentation = Segmentation10x(
                    input_file=input_file,
                    output_dir=segmentation_dir
                )
            else:
                logger.info(f"Processing 40X file: {input_file}")
                segmentation = Segmentation40x(
                    input_file=input_file,
                    output_dir=segmentation_dir
                )
            
            # Run segmentation for specified wells
            segmentation.run(wells=wells)
            
        except Exception as e:
            logger.error(f"Error in segmentation for {input_file}: {e}")
            success = False
    
    return success

def run_mapping(segmentation_dir, output_dir, wells, logger):
    """Run mapping step of the pipeline.
    
    Args:
        segmentation_dir: Directory containing segmentation results
        output_dir: Output directory
        wells: List of wells to process
        logger: Logger instance
        
    Returns:
        bool: True if successful, False otherwise
    """
    from imageanalysis.core.mapping.pipeline import MappingPipeline
    
    logger.info("Running mapping...")
    mapping_dir = os.path.join(output_dir, "mapping")
    os.makedirs(mapping_dir, exist_ok=True)
    
    # Set up directories for 10X and 40X data
    seg_10x_dir = os.path.join(segmentation_dir, "10x")
    seg_40x_dir = os.path.join(segmentation_dir, "40x")
    
    # If directories don't exist, assume all data is in the main directory
    if not os.path.exists(seg_10x_dir) or not os.path.exists(seg_40x_dir):
        logger.info("Using single directory for both magnifications")
        seg_10x_dir = segmentation_dir
        seg_40x_dir = segmentation_dir
    
    try:
        # Create and run mapping pipeline
        pipeline = MappingPipeline(
            seg_10x_dir=seg_10x_dir,
            seg_40x_dir=seg_40x_dir,
            output_dir=mapping_dir,
            config={
                'matching': {
                    'max_iterations': 5,
                    'distance_threshold': 50.0,
                    'ransac_threshold': 20.0
                }
            }
        )
        
        results = pipeline.run(wells=wells)
        
        # Print summary
        logger.info("\nMapping Results Summary:")
        for well, result in results.items():
            logger.info(f"\nWell {well}:")
            logger.info(f"  Matched points: {len(result.matched_points_10x)}")
            logger.info(f"  RMSE: {result.error_metrics['rmse']:.2f} pixels")
            logger.info(f"  Max error: {result.error_metrics['max_error']:.2f} pixels")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in mapping: {e}")
        return False

def run_genotyping(input_files, segmentation_dir, output_dir, wells, logger):
    """Run genotyping step of the pipeline.
    
    Args:
        input_files: List of input ND2 files
        segmentation_dir: Directory containing segmentation results
        output_dir: Output directory
        wells: List of wells to process
        logger: Logger instance
        
    Returns:
        bool: True if successful, False otherwise
    """
    from imageanalysis.core.genotyping.pipeline import GenotypingPipeline
    
    logger.info("Running genotyping...")
    genotyping_dir = os.path.join(output_dir, "genotyping")
    os.makedirs(genotyping_dir, exist_ok=True)
    
    # Create a temporary barcode library file if not available
    barcode_file = None
    
    try:
        # Create a simple barcode library for testing
        barcode_file = os.path.join(tempfile.gettempdir(), "test_barcodes.csv")
        with open(barcode_file, "w") as f:
            f.write("Barcode,Gene\n")
            f.write("ACGTACGT,Gene1\n")
            f.write("TGCATGCA,Gene2\n")
            f.write("AATTCCGG,Gene3\n")
        
        # Process genotyping images
        for input_file in input_files:
            try:
                logger.info(f"Processing genotyping for {input_file}")
                
                pipeline = GenotypingPipeline(
                    input_file=input_file,
                    segmentation_dir=segmentation_dir,
                    output_dir=genotyping_dir,
                    barcode_library=barcode_file
                )
                
                pipeline.run(wells=wells)
                
            except Exception as e:
                logger.error(f"Error in genotyping for {input_file}: {e}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error in genotyping setup: {e}")
        return False
    finally:
        # Clean up temporary file if created
        if barcode_file and os.path.exists(barcode_file):
            os.remove(barcode_file)

def run_phenotyping(input_files, segmentation_dir, genotyping_dir, output_dir, wells, logger):
    """Run phenotyping step of the pipeline.
    
    Args:
        input_files: List of input ND2 files
        segmentation_dir: Directory containing segmentation results
        genotyping_dir: Directory containing genotyping results
        output_dir: Output directory
        wells: List of wells to process
        logger: Logger instance
        
    Returns:
        bool: True if successful, False otherwise
    """
    from imageanalysis.core.phenotyping.pipeline import PhenotypingPipeline
    
    logger.info("Running phenotyping...")
    phenotyping_dir = os.path.join(output_dir, "phenotyping")
    os.makedirs(phenotyping_dir, exist_ok=True)
    
    try:
        # Process phenotyping images
        for input_file in input_files:
            try:
                logger.info(f"Processing phenotyping for {input_file}")
                
                pipeline = PhenotypingPipeline(
                    input_file=input_file,
                    segmentation_dir=segmentation_dir,
                    genotyping_dir=genotyping_dir,
                    output_dir=phenotyping_dir,
                    channels=["DAPI", "mClov3", "TMR"]
                )
                
                pipeline.run(wells=wells)
                
            except Exception as e:
                logger.error(f"Error in phenotyping for {input_file}: {e}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error in phenotyping: {e}")
        return False

def create_albums(phenotyping_dir, output_dir, wells, logger):
    """Create album visualizations from phenotyping results.
    
    Args:
        phenotyping_dir: Directory containing phenotyping results
        output_dir: Output directory
        wells: List of wells to process
        logger: Logger instance
        
    Returns:
        bool: True if successful, False otherwise
    """
    from imageanalysis.core.visualization.albums import AlbumCreator
    
    logger.info("Creating albums...")
    albums_dir = os.path.join(output_dir, "albums")
    os.makedirs(albums_dir, exist_ok=True)
    
    try:
        # Create albums
        creator = AlbumCreator(
            phenotyping_dir=phenotyping_dir,
            output_dir=albums_dir
        )
        
        creator.run(wells=wells)
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating albums: {e}")
        return False

def generate_report(output_dir, results, logger):
    """Generate a report summarizing the test results.
    
    Args:
        output_dir: Output directory
        results: Dictionary of test results
        logger: Logger instance
    """
    logger.info("Generating test report...")
    report_dir = os.path.join(output_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    
    # Create report
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(report_dir, f"test_report_{timestamp}.json")
    
    report = {
        "timestamp": timestamp,
        "results": results,
        "success": all(results.values())
    }
    
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Report saved to {report_file}")
    
    # Print summary
    logger.info("\nTest Summary:")
    for step, success in results.items():
        status = "PASSED" if success else "FAILED"
        logger.info(f"  {step}: {status}")
    
    overall = "PASSED" if all(results.values()) else "FAILED"
    logger.info(f"\nOverall Test: {overall}")

def find_nd2_files(directory):
    """Find all ND2 files in the given directory and its subdirectories.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of ND2 file paths
    """
    nd2_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".nd2"):
                nd2_files.append(os.path.join(root, file))
    
    return nd2_files

def get_test_data(data_dir, logger):
    """Find or create test data for running the pipeline.
    
    Args:
        data_dir: Base directory for data
        logger: Logger instance
        
    Returns:
        List of input files to process
    """
    # Look for ND2 files in the data directory
    nd2_files = find_nd2_files(data_dir)
    
    if nd2_files:
        logger.info(f"Found {len(nd2_files)} ND2 files in {data_dir}")
        return nd2_files
    
    # If no ND2 files found, check if we already have segmentation results
    # that we can use for downstream steps
    seg_dir = os.path.join(data_dir, "new", "segmentation")
    if os.path.exists(seg_dir):
        logger.info(f"No ND2 files found, but segmentation results exist in {seg_dir}")
        logger.info("Will skip segmentation and use existing results for downstream steps")
        return []
    
    logger.error("No ND2 files or segmentation results found in data directory")
    return []

def main():
    """Main function."""
    args = parse_args()
    
    # Set up logging
    logger = setup_logging(args.output_dir, args.debug)
    
    logger.info("=== Full Pipeline Test ===")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Wells to process: {args.wells}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find test data
    input_files = get_test_data(args.data_dir, logger)
    
    # Track results
    results = {}
    
    # Run pipeline steps
    try:
        # Paths to output directories
        seg_dir = os.path.join(args.output_dir, "segmentation")
        map_dir = os.path.join(args.output_dir, "mapping")
        geno_dir = os.path.join(args.output_dir, "genotyping")
        pheno_dir = os.path.join(args.output_dir, "phenotyping")
        
        # If we have ND2 files, run segmentation
        if input_files:
            results["segmentation"] = run_segmentation(input_files, args.output_dir, args.wells, logger)
        else:
            # Use existing segmentation results
            seg_dir = os.path.join(args.data_dir, "new", "segmentation")
            logger.info(f"Using existing segmentation results from {seg_dir}")
            results["segmentation"] = True
        
        # Run mapping
        results["mapping"] = run_mapping(seg_dir, args.output_dir, args.wells, logger)
        
        # Run genotyping if we have input files
        if input_files:
            results["genotyping"] = run_genotyping(input_files, seg_dir, args.output_dir, args.wells, logger)
        else:
            # Use existing genotyping results
            geno_dir = os.path.join(args.data_dir, "new", "genotyping")
            logger.info(f"Using existing genotyping results from {geno_dir}")
            results["genotyping"] = True
        
        # Run phenotyping
        if input_files:
            results["phenotyping"] = run_phenotyping(input_files, seg_dir, geno_dir, args.output_dir, args.wells, logger)
        else:
            # Use existing phenotyping results
            pheno_dir = os.path.join(args.data_dir, "new", "phenotyping")
            logger.info(f"Using existing phenotyping results from {pheno_dir}")
            results["phenotyping"] = True
        
        # Create albums
        results["albums"] = create_albums(pheno_dir, args.output_dir, args.wells, logger)
        
        # Generate report
        generate_report(args.output_dir, results, logger)
        
        return 0 if all(results.values()) else 1
        
    except Exception as e:
        logger.error(f"Error in full pipeline test: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())