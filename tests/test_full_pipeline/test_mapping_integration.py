#!/usr/bin/env python3
"""Test script for the mapping component integration with the full pipeline."""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
import time

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test mapping integration with the full pipeline"
    )
    
    parser.add_argument(
        "--seg-dir",
        default="/home/casey/Desktop/ShalemLab/ImageAnalysis/results/production/new/segmentation",
        help="Directory containing segmentation results"
    )
    
    parser.add_argument(
        "--centroids-dir",
        default="/home/casey/Desktop/ShalemLab/ImageAnalysis/results/production/new/centroids",
        help="Directory containing centroids"
    )
    
    parser.add_argument(
        "--output-dir",
        default="/home/casey/Desktop/ShalemLab/ImageAnalysis/results/mapping_integration_test",
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
    """Set up logging for the test script."""
    # Create log directory
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logger
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"mapping_integration_test_{timestamp}.log")
    
    logger = logging.getLogger("mapping_integration_test")
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

def run_mapping(centroids_dir, output_dir, wells, logger):
    """Run mapping step of the pipeline."""
    from imageanalysis.core.mapping.pipeline import MappingPipeline
    
    logger.info("Running mapping...")
    mapping_dir = os.path.join(output_dir, "mapping")
    os.makedirs(mapping_dir, exist_ok=True)
    
    try:
        # Create and run mapping pipeline
        pipeline = MappingPipeline(
            seg_10x_dir=centroids_dir,
            seg_40x_dir=centroids_dir,
            output_dir=mapping_dir,
            config={
                'matching': {
                    'max_iterations': 5,
                    'distance_threshold': 100.0,
                    'ransac_threshold': 30.0
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
            
            # Save more detailed results
            detailed_results = {
                'transform': {
                    'dx': result.transform.dx,
                    'dy': result.transform.dy,
                    'theta': result.transform.theta,
                    'scale_x': result.transform.scale_x,
                    'scale_y': result.transform.scale_y
                },
                'error_metrics': result.error_metrics,
                'num_points': len(result.matched_points_10x)
            }
            
            detailed_file = os.path.join(mapping_dir, f"{well}_detailed_results.json")
            with open(detailed_file, 'w') as f:
                json.dump(detailed_results, f, indent=2)
            
            logger.info(f"  Detailed results saved to: {detailed_file}")
        
        return True, results
        
    except Exception as e:
        logger.error(f"Error in mapping: {e}")
        return False, {}

def main():
    """Main function."""
    args = parse_args()
    
    # Set up logging
    logger = setup_logging(args.output_dir, args.debug)
    
    logger.info("=== Mapping Integration Test ===")
    logger.info(f"Segmentation directory: {args.seg_dir}")
    logger.info(f"Centroids directory: {args.centroids_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Wells to process: {args.wells}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check that centroids exist
    for well in args.wells:
        centroid_10x_file = os.path.join(args.centroids_dir, well, f"{well}_nuclei_centroids.npy")
        centroid_40x_file = os.path.join(args.centroids_dir, well, f"{well}_nuclei_centroids_40x.npy")
        
        if not os.path.exists(centroid_10x_file):
            logger.error(f"10X centroids file not found: {centroid_10x_file}")
            return 1
        
        if not os.path.exists(centroid_40x_file):
            logger.error(f"40X centroids file not found: {centroid_40x_file}")
            return 1
    
    # Run mapping
    success, results = run_mapping(args.centroids_dir, args.output_dir, args.wells, logger)
    
    # Generate report
    report_dir = os.path.join(args.output_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(report_dir, f"mapping_report_{timestamp}.json")
    
    report = {
        "timestamp": timestamp,
        "success": success,
        "wells_processed": len(results)
    }
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\nReport saved to: {report_file}")
    logger.info(f"Overall test {'PASSED' if success else 'FAILED'}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())