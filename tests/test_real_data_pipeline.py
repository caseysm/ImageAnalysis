#!/usr/bin/env python3
"""
Test the full image analysis pipeline using real data.

This script runs the image analysis pipeline on the actual data files
in the data directory and verifies the results.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import json
import shutil
import logging
import time
from datetime import datetime
from pathlib import Path

# Import from the imageanalysis package
from imageanalysis.core.segmentation import Segmentation10XPipeline
from imageanalysis.core.genotyping.pipeline import StandardGenotypingPipeline
from imageanalysis.core.phenotyping.pipeline import PhenotypingPipeline
from imageanalysis.utils.logging import setup_logger

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the full pipeline on real data"
    )
    
    parser.add_argument(
        "--output-dir",
        default="results/real_data_test",
        help="Directory to store test results"
    )
    
    parser.add_argument(
        "--limit-files",
        type=int,
        default=5,
        help="Limit the number of files to process per well (default: 5, use -1 for no limit)"
    )
    
    parser.add_argument(
        "--wells",
        nargs="+",
        default=["Well1"],
        help="Wells to process (default: Well1, use --wells all to process all wells)"
    )
    
    parser.add_argument(
        "--all-data",
        action="store_true",
        help="Process all data without limits (equivalent to --limit-files -1 --wells all)"
    )
    
    parser.add_argument(
        "--max-files-per-stage",
        type=int,
        default=10,
        help="Maximum number of files to process per pipeline stage (default: 10)"
    )
    
    return parser.parse_args()

def set_up_test_directory(base_dir):
    """Set up the test directory structure."""
    directories = [
        os.path.join(base_dir, "segmentation"),
        os.path.join(base_dir, "genotyping"),
        os.path.join(base_dir, "phenotyping"),
        os.path.join(base_dir, "albums"),
        os.path.join(base_dir, "logs")
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    log_file = os.path.join(base_dir, "logs", f"full_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("real_data_test")

def find_data_files(data_type, well_id, limit=5):
    """Find data files for a specific well and data type.
    
    Args:
        data_type: Type of data (phenotyping, genotyping)
        well_id: ID of the well to look for
        limit: Maximum number of files to return, -1 for no limit
        
    Returns:
        List of matching file paths
    """
    data_dir = os.path.join("data", data_type)
    
    if not os.path.exists(data_dir):
        logging.error(f"Data directory not found: {data_dir}")
        return []
    
    if data_type == "genotyping":
        # For genotyping, look in all cycle directories
        cycle_dirs = [d for d in os.listdir(data_dir) if d.startswith('cycle_')]
        files = []
        for cycle_dir in cycle_dirs:
            cycle_path = os.path.join(data_dir, cycle_dir)
            if os.path.isdir(cycle_path):
                well_files = [
                    os.path.join(cycle_path, f) for f in os.listdir(cycle_path)
                    if f.startswith(well_id) and f.endswith(".nd2")
                ]
                # Apply limit if set (per cycle directory)
                if limit > 0:
                    well_files = well_files[:limit]
                files.extend(well_files)
        return files
    else:
        # For other data types, look directly in the data directory
        files = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.startswith(well_id) and f.endswith(".nd2")
        ]
        # Apply limit if set
        if limit > 0:
            return files[:limit]
        return files

def run_segmentation(data_files, output_dir, well_id, logger, args):
    """Run the segmentation pipeline on a set of files."""
    logger.info(f"Running segmentation for {well_id} with {len(data_files)} files")
    
    if not data_files:
        logger.error("No data files found for segmentation")
        return False
    
    # Set up counter for processed files
    processed_files = 0
    segmentation_dir = os.path.join(output_dir, "segmentation")
    success = True
    start_time = time.time()
    
    # Process files up to the specified maximum
    max_files = min(args.max_files_per_stage, len(data_files))
    logger.info(f"Will process {max_files} files for segmentation")
    
    for i, input_file in enumerate(data_files[:max_files]):
        logger.info(f"Segmenting file {i+1}/{max_files}: {os.path.basename(input_file)}")
        
        # Create segmentation pipeline config
        config = {
            'input_file': input_file,
            'output_dir': segmentation_dir,
            'wells': [well_id],
            'nuclear_channel': 0,  # Assuming DAPI is first channel
            'cell_channel': 1      # Assuming cell stain is second channel
        }
        
        try:
            # Run segmentation pipeline
            pipeline = Segmentation10XPipeline(config)
            results = pipeline.run()
            
            logger.info(f"Segmentation completed for file {i+1}: {results}")
            processed_files += 1
        except Exception as e:
            logger.error(f"Segmentation failed for file {i+1}: {str(e)}")
            success = False
    
    elapsed = time.time() - start_time
    logger.info(f"Segmentation of {processed_files} files completed in {elapsed:.2f} seconds")
    
    return success and processed_files > 0

def run_genotyping(data_files, output_dir, well_id, logger, args):
    """Run the genotyping pipeline on a set of files."""
    logger.info(f"Running genotyping for {well_id} with {len(data_files)} files")
    
    if not data_files:
        logger.error("No data files found for genotyping")
        return False
    
    # Create path to segmentation results
    segmentation_dir = os.path.join(output_dir, "segmentation")
    
    # Check if segmentation results exist
    if not os.path.exists(os.path.join(segmentation_dir, well_id)):
        logger.error(f"No segmentation results found for {well_id}")
        return False
    
    # Locate or create a dummy barcode library
    barcode_library = os.path.join("data", "RBP_F2_Bulk_Optimized_Final.csv")
    if not os.path.exists(barcode_library):
        logger.warning("Barcode library not found, creating a dummy one")
        # Create a dummy barcode library
        df = pd.DataFrame({
            'barcode': ['ACGT', 'GACT', 'TGCA', 'CATG', 'GTAC'],
            'gene': ['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5']
        })
        barcode_library = os.path.join(output_dir, "dummy_barcodes.csv")
        df.to_csv(barcode_library, index=False)
    
    # Set up counter for processed files
    processed_files = 0
    genotyping_dir = os.path.join(output_dir, "genotyping")
    success = True
    start_time = time.time()
    
    # Group files by cycle
    cycle_files = {}
    for file_path in data_files:
        # Extract cycle from file path
        # Assuming directory structure like data/genotyping/cycle_N/file.nd2
        parts = file_path.split(os.sep)
        if len(parts) >= 3 and parts[-2].startswith("cycle_"):
            cycle = parts[-2]
            if cycle not in cycle_files:
                cycle_files[cycle] = []
            cycle_files[cycle].append(file_path)
    
    # Process one file from each cycle (to be comprehensive)
    for cycle, files in cycle_files.items():
        if not files:
            continue
            
        # Take first file from each cycle
        input_file = files[0]
        logger.info(f"Genotyping cycle {cycle}: {os.path.basename(input_file)}")
        
        # Create genotyping pipeline config
        config = {
            'input_file': input_file,
            'segmentation_dir': segmentation_dir,
            'barcode_library': barcode_library,
            'output_dir': genotyping_dir,
            'wells': [well_id],
            'min_quality_score': 0.7,
            'max_hamming_distance': 1
        }
        
        try:
            # Run genotyping pipeline
            pipeline = StandardGenotypingPipeline(config)
            results = pipeline.run()
            
            logger.info(f"Genotyping completed for {cycle}: {results}")
            processed_files += 1
        except Exception as e:
            logger.error(f"Genotyping failed for {cycle}: {str(e)}")
            success = False
    
    elapsed = time.time() - start_time
    logger.info(f"Genotyping of {processed_files} cycles completed in {elapsed:.2f} seconds")
    
    return success and processed_files > 0

def run_phenotyping(data_files, output_dir, well_id, logger, args):
    """Run the phenotyping pipeline on a set of files."""
    logger.info(f"Running phenotyping for {well_id} with {len(data_files)} files")
    
    if not data_files:
        logger.error("No data files found for phenotyping")
        return False
    
    # Create paths to required directories
    segmentation_dir = os.path.join(output_dir, "segmentation")
    genotyping_dir = os.path.join(output_dir, "genotyping")
    
    # Check if segmentation results exist
    if not os.path.exists(os.path.join(segmentation_dir, well_id)):
        logger.error(f"No segmentation results found for {well_id}")
        return False
    
    # Set up counter for processed files
    processed_files = 0
    phenotyping_dir = os.path.join(output_dir, "phenotyping")
    success = True
    start_time = time.time()
    
    # Process files up to the specified maximum
    max_files = min(args.max_files_per_stage, len(data_files))
    logger.info(f"Will process {max_files} files for phenotyping")
    
    for i, input_file in enumerate(data_files[:max_files]):
        logger.info(f"Phenotyping file {i+1}/{max_files}: {os.path.basename(input_file)}")
        
        # Create phenotyping pipeline config
        config = {
            'input_file': input_file,
            'segmentation_dir': segmentation_dir,
            'genotyping_dir': genotyping_dir,
            'output_dir': phenotyping_dir,
            'wells': [well_id],
            'channels': ['DAPI', 'mClov3', 'TMR']  # Assume standard channels
        }
        
        try:
            # Run phenotyping pipeline
            pipeline = PhenotypingPipeline(config)
            results = pipeline.run()
            
            logger.info(f"Phenotyping completed for file {i+1}: {results}")
            processed_files += 1
        except Exception as e:
            logger.error(f"Phenotyping failed for file {i+1}: {str(e)}")
            success = False
    
    elapsed = time.time() - start_time
    logger.info(f"Phenotyping of {processed_files} files completed in {elapsed:.2f} seconds")
    
    return success and processed_files > 0

def verify_results(output_dir, well_id, logger):
    """Verify that the pipeline results are as expected."""
    success = True
    
    # Check segmentation results
    segmentation_dir = os.path.join(output_dir, "segmentation", well_id)
    if not os.path.exists(segmentation_dir):
        logger.error(f"Segmentation directory not found: {segmentation_dir}")
        success = False
    else:
        # Check for mask files
        mask_files = [f for f in os.listdir(segmentation_dir) if f.endswith("_mask.npy")]
        if not mask_files:
            logger.error("No mask files found in segmentation results")
            success = False
        else:
            logger.info(f"Found {len(mask_files)} mask files in segmentation results")
            
            # Check one mask file
            mask_file = os.path.join(segmentation_dir, mask_files[0])
            try:
                mask = np.load(mask_file)
                logger.info(f"Loaded mask with shape {mask.shape}, max value {np.max(mask)}")
            except Exception as e:
                logger.error(f"Failed to load mask file: {str(e)}")
                success = False
    
    # Check genotyping results
    genotyping_dir = os.path.join(output_dir, "genotyping", well_id)
    if not os.path.exists(genotyping_dir):
        logger.error(f"Genotyping directory not found: {genotyping_dir}")
        success = False
    else:
        # Check for genotype file
        genotype_file = os.path.join(genotyping_dir, f"{well_id}_genotypes.csv")
        if not os.path.exists(genotype_file):
            logger.error(f"Genotype file not found: {genotype_file}")
            success = False
        else:
            try:
                genotypes = pd.read_csv(genotype_file)
                logger.info(f"Loaded genotypes file with {len(genotypes)} rows")
            except Exception as e:
                logger.error(f"Failed to load genotypes file: {str(e)}")
                success = False
    
    # Check phenotyping results
    phenotyping_dir = os.path.join(output_dir, "phenotyping", well_id)
    if not os.path.exists(phenotyping_dir):
        logger.error(f"Phenotyping directory not found: {phenotyping_dir}")
        success = False
    else:
        # Check for phenotype file
        phenotype_file = os.path.join(phenotyping_dir, f"{well_id}_phenotypes.csv")
        if not os.path.exists(phenotype_file):
            logger.error(f"Phenotype file not found: {phenotype_file}")
            success = False
        else:
            try:
                phenotypes = pd.read_csv(phenotype_file)
                logger.info(f"Loaded phenotypes file with {len(phenotypes)} rows")
            except Exception as e:
                logger.error(f"Failed to load phenotypes file: {str(e)}")
                success = False
    
    return success

def run_full_pipeline(args):
    """Run the full pipeline on real data."""
    # Set up test directory
    output_dir = args.output_dir
    logger = set_up_test_directory(output_dir)
    
    # Handle the all-data option
    if args.all_data:
        args.limit_files = -1
        args.wells = ["all"]
    
    # Handle "all" wells by finding all available wells
    if "all" in args.wells:
        # Find all wells in the phenotyping directory
        phenotyping_dir = os.path.join("data", "phenotyping")
        if os.path.exists(phenotyping_dir):
            well_pattern = r"(Well\d+)_"
            import re
            well_ids = set()
            for filename in os.listdir(phenotyping_dir):
                if filename.endswith(".nd2"):
                    match = re.match(well_pattern, filename)
                    if match:
                        well_ids.add(match.group(1))
            args.wells = list(well_ids) if well_ids else ["Well1"]
    
    logger.info(f"Starting full pipeline test with real data")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Wells to process: {args.wells}")
    
    # Show limit info
    if args.limit_files < 0:
        logger.info("Processing all files (no limit)")
    else:
        logger.info(f"Limiting to {args.limit_files} files per well")
    
    start_time = time.time()
    success = True
    
    # Create a test report
    report = {
        "timestamp": datetime.now().isoformat(),
        "wells": args.wells,
        "limit_files": args.limit_files,
        "results": {}
    }
    
    # Process each well
    for well_id in args.wells:
        logger.info(f"Processing well: {well_id}")
        well_start_time = time.time()
        
        # Find data files
        phenotyping_files = find_data_files("phenotyping", well_id, args.limit_files)
        genotyping_files = find_data_files("genotyping", well_id, args.limit_files)
        
        logger.info(f"Found {len(phenotyping_files)} phenotyping files and {len(genotyping_files)} genotyping files")
        
        # Run segmentation on phenotyping files
        seg_success = run_segmentation(phenotyping_files, output_dir, well_id, logger, args)
        
        # Run genotyping on genotyping files
        if seg_success:
            geno_success = run_genotyping(genotyping_files, output_dir, well_id, logger, args)
        else:
            geno_success = False
            logger.error("Skipping genotyping due to segmentation failure")
        
        # Run phenotyping on phenotyping files
        if seg_success:
            pheno_success = run_phenotyping(phenotyping_files, output_dir, well_id, logger, args)
        else:
            pheno_success = False
            logger.error("Skipping phenotyping due to segmentation failure")
        
        # Verify results
        verification = verify_results(output_dir, well_id, logger)
        
        # Update success flag
        well_success = seg_success and geno_success and pheno_success and verification
        success = success and well_success
        
        # Record well results
        well_time = time.time() - well_start_time
        report["results"][well_id] = {
            "success": well_success,
            "segmentation": seg_success,
            "genotyping": geno_success,
            "phenotyping": pheno_success,
            "verification": verification,
            "time_seconds": well_time
        }
        
        logger.info(f"Completed processing well {well_id} in {well_time:.2f} seconds - " +
                   ("SUCCESS" if well_success else "FAILURE"))
    
    # Finalize report
    total_time = time.time() - start_time
    report["total_time_seconds"] = total_time
    report["overall_success"] = success
    
    # Save report
    report_dir = os.path.join(output_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    
    report_file = os.path.join(report_dir, f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Completed full pipeline test in {total_time:.2f} seconds - " +
               ("SUCCESS" if success else "FAILURE"))
    logger.info(f"Test report saved to {report_file}")
    
    return success

def main():
    """Main entry point."""
    args = parse_args()
    success = run_full_pipeline(args)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())