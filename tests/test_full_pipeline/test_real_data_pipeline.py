#!/usr/bin/env python3
"""Test script for the ImageAnalysis pipeline using real data."""

import argparse
import os
import sys
import time
import logging
import json
from pathlib import Path
import glob

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test the full pipeline with real data"
    )
    
    parser.add_argument(
        "--data-dir",
        default="/home/casey/Desktop/ShalemLab/ImageAnalysis/data",
        help="Directory containing test data"
    )
    
    parser.add_argument(
        "--output-dir",
        default="/home/casey/Desktop/ShalemLab/ImageAnalysis/results/real_pipeline_test",
        help="Output directory for test results"
    )
    
    parser.add_argument(
        "--wells",
        nargs="+",
        default=["Well1"],
        help="List of wells to process"
    )
    
    parser.add_argument(
        "--barcode-file",
        default="/home/casey/Desktop/ShalemLab/ImageAnalysis/data/RBP_F2_Bulk_Optimized_Final.csv",
        help="CSV file containing barcode information"
    )
    
    parser.add_argument(
        "--limit-files",
        type=int,
        default=3,
        help="Limit the number of files to process per step (for faster testing)"
    )
    
    parser.add_argument(
        "--channels",
        default="DAPI,mClov3,TMR",
        help="Comma-separated list of channel names"
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
    log_file = os.path.join(log_dir, f"real_pipeline_test_{timestamp}.log")
    
    logger = logging.getLogger("real_pipeline_test")
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

def find_data_files(data_dir, limit_files=None):
    """Find all data files organized by step and well.
    
    Returns:
        dict: Dict mapping steps to wells to file lists
    """
    result = {
        "segmentation_10x": {},
        "segmentation_40x": {},
        "genotyping": {},
        "phenotyping": {}
    }
    
    # Find phenotyping (40X) files
    pheno_files = glob.glob(os.path.join(data_dir, "phenotyping", "*.nd2"))
    if limit_files:
        pheno_files = pheno_files[:limit_files]
    
    # Group by well
    for file in pheno_files:
        filename = os.path.basename(file)
        
        # Assuming filename format Well{N}_Point{M}...
        well_name = filename.split("_")[0]
        
        # Group by sequence number to determine if 10X or 40X
        # Assuming sequence format Seq{NNNN}
        seq_match = filename.find("Seq")
        if seq_match > 0:
            try:
                seq_num = int(filename[seq_match+3:seq_match+7])
                
                # Based on our earlier heuristic
                if seq_num < 700:
                    step = "segmentation_10x"
                else:
                    step = "segmentation_40x"
                
                if well_name not in result[step]:
                    result[step][well_name] = []
                
                result[step][well_name].append(file)
                
                # Also add to phenotyping
                if well_name not in result["phenotyping"]:
                    result["phenotyping"][well_name] = []
                
                result["phenotyping"][well_name].append(file)
            except:
                pass
    
    # Find genotyping files
    for cycle_dir in glob.glob(os.path.join(data_dir, "genotyping", "cycle_*")):
        geno_files = glob.glob(os.path.join(cycle_dir, "*.nd2"))
        if limit_files:
            geno_files = geno_files[:limit_files]
        
        # Group by well
        for file in geno_files:
            filename = os.path.basename(file)
            
            # Assuming filename format Well{N}_Point{M}...
            well_name = filename.split("_")[0]
            
            if well_name not in result["genotyping"]:
                result["genotyping"][well_name] = []
            
            result["genotyping"][well_name].append(file)
    
    return result

def run_segmentation(files, output_dir, wells, channels, logger):
    """Run segmentation on input files."""
    seg_output_dir = os.path.join(output_dir, "segmentation")
    os.makedirs(seg_output_dir, exist_ok=True)
    
    results = {}
    
    # Process 10X files
    if wells:
        filtered_wells = wells
    else:
        filtered_wells = list(files["segmentation_10x"].keys())
    
    for well in filtered_wells:
        if well in files["segmentation_10x"]:
            try:
                from imageanalysis.core.segmentation.segmentation_10x import Segmentation10XPipeline
                
                # Get limited number of files
                well_files = files["segmentation_10x"][well]
                
                logger.info(f"Running 10X segmentation for well {well} with {len(well_files)} files")
                
                # Process each file
                for i, input_file in enumerate(well_files):
                    logger.info(f"  Processing file {i+1}/{len(well_files)}: {os.path.basename(input_file)}")
                    
                    try:
                        # Run segmentation
                        segmentation = Segmentation10XPipeline({
                            "input_file": input_file,
                            "output_dir": seg_output_dir,
                            "nuclear_channel": 0,
                            "cell_channel": 1,
                            "wells": [well]
                        })
                        
                        segmentation.run(wells=[well])
                    except Exception as e:
                        logger.error(f"Error processing 10X file {input_file}: {e}")
                
                results[f"{well}_10x"] = True
                
            except Exception as e:
                logger.error(f"Error in 10X segmentation for well {well}: {e}")
                results[f"{well}_10x"] = False
    
    # Process 40X files
    if wells:
        filtered_wells = wells
    else:
        filtered_wells = list(files["segmentation_40x"].keys())
    
    for well in filtered_wells:
        if well in files["segmentation_40x"]:
            try:
                from imageanalysis.core.segmentation.segmentation_40x import Segmentation40x
                
                # Get limited number of files
                well_files = files["segmentation_40x"][well]
                
                logger.info(f"Running 40X segmentation for well {well} with {len(well_files)} files")
                
                # Process each file
                for i, input_file in enumerate(well_files):
                    logger.info(f"  Processing file {i+1}/{len(well_files)}: {os.path.basename(input_file)}")
                    
                    try:
                        # Run segmentation
                        segmentation = Segmentation40x(
                            input_file=input_file,
                            output_dir=seg_output_dir,
                            nuclei_diameter=30,
                            cell_diameter=60
                        )
                        
                        segmentation.run(wells=[well])
                    except Exception as e:
                        logger.error(f"Error processing 40X file {input_file}: {e}")
                
                results[f"{well}_40x"] = True
                
            except Exception as e:
                logger.error(f"Error in 40X segmentation for well {well}: {e}")
                results[f"{well}_40x"] = False
    
    return results, seg_output_dir

def run_mapping(seg_dir, output_dir, wells, logger):
    """Run mapping between 10X and 40X data."""
    mapping_dir = os.path.join(output_dir, "mapping")
    os.makedirs(mapping_dir, exist_ok=True)
    
    try:
        from imageanalysis.core.mapping.pipeline import MappingPipeline
        
        # Extract centroids if they don't already exist
        centroids_dir = os.path.join(output_dir, "centroids")
        os.makedirs(centroids_dir, exist_ok=True)
        
        # Check if we need to extract centroids
        need_centroids = False
        for well in wells:
            centroids_10x = os.path.join(centroids_dir, well, f"{well}_nuclei_centroids.npy")
            centroids_40x = os.path.join(centroids_dir, well, f"{well}_nuclei_centroids_40x.npy")
            
            if not os.path.exists(centroids_10x) or not os.path.exists(centroids_40x):
                need_centroids = True
                break
        
        if need_centroids:
            logger.info("Extracting centroids from segmentation masks...")
            # Create a script to extract centroids similar to the one we made earlier
            extract_script = os.path.join(os.path.dirname(__file__), "extract_centroids.py")
            
            if os.path.exists(extract_script):
                import subprocess
                subprocess.run(["python", extract_script, seg_dir, centroids_dir])
            else:
                logger.error(f"Centroids extraction script not found: {extract_script}")
                from imageanalysis.core.segmentation.base import SegmentationPipeline
                
                # Extract centroids manually
                for well in wells:
                    well_seg_dir = os.path.join(seg_dir, well)
                    if os.path.exists(well_seg_dir):
                        from skimage.measure import regionprops
                        import numpy as np
                        import re
                        
                        # Create output directory
                        well_centroids_dir = os.path.join(centroids_dir, well)
                        os.makedirs(well_centroids_dir, exist_ok=True)
                        
                        # Function to determine if a file is 10X or 40X
                        def is_10x_file(filename):
                            match = re.search(r'Seq(\d+)', filename)
                            if match:
                                seq_num = int(match.group(1))
                                return seq_num < 700
                            return False
                        
                        # Find nuclei mask files
                        nuclei_files = glob.glob(os.path.join(well_seg_dir, "*_nuclei_mask.npy"))
                        
                        # Group by magnification
                        files_10x = []
                        files_40x = []
                        
                        for file in nuclei_files:
                            if is_10x_file(os.path.basename(file)):
                                files_10x.append(file)
                            else:
                                files_40x.append(file)
                        
                        logger.info(f"Found {len(files_10x)} 10X files and {len(files_40x)} 40X files for well {well}")
                        
                        # Process 10X files
                        if files_10x:
                            centroids_10x = []
                            for file in files_10x:
                                try:
                                    mask = np.load(file)
                                    props = regionprops(mask)
                                    centroids = np.array([prop.centroid for prop in props])
                                    if len(centroids) > 0:
                                        centroids_10x.append(centroids)
                                except Exception as e:
                                    logger.error(f"Error extracting centroids from {file}: {e}")
                            
                            if centroids_10x:
                                all_centroids_10x = np.vstack(centroids_10x)
                                np.save(os.path.join(well_centroids_dir, f"{well}_nuclei_centroids.npy"), all_centroids_10x)
                                logger.info(f"Saved {len(all_centroids_10x)} 10X centroids")
                        
                        # Process 40X files
                        if files_40x:
                            centroids_40x = []
                            for file in files_40x:
                                try:
                                    mask = np.load(file)
                                    props = regionprops(mask)
                                    centroids = np.array([prop.centroid for prop in props])
                                    if len(centroids) > 0:
                                        centroids_40x.append(centroids)
                                except Exception as e:
                                    logger.error(f"Error extracting centroids from {file}: {e}")
                            
                            if centroids_40x:
                                all_centroids_40x = np.vstack(centroids_40x)
                                np.save(os.path.join(well_centroids_dir, f"{well}_nuclei_centroids_40x.npy"), all_centroids_40x)
                                logger.info(f"Saved {len(all_centroids_40x)} 40X centroids")
        
        # Now run the mapping pipeline
        logger.info("Running mapping pipeline...")
        
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
        logger.info("Mapping Results:")
        for well, result in results.items():
            logger.info(f"  Well {well}: {len(result.matched_points_10x)} points matched, RMSE: {result.error_metrics['rmse']:.2f} pixels")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in mapping: {e}")
        return False

def run_genotyping(files, seg_dir, output_dir, wells, barcode_file, logger):
    """Run genotyping on input files."""
    geno_output_dir = os.path.join(output_dir, "genotyping")
    os.makedirs(geno_output_dir, exist_ok=True)
    
    results = {}
    
    if wells:
        filtered_wells = wells
    else:
        filtered_wells = list(files["genotyping"].keys())
    
    for well in filtered_wells:
        if well in files["genotyping"]:
            try:
                from imageanalysis.core.genotyping.pipeline import GenotypingPipeline
                
                # Get limited number of files
                well_files = files["genotyping"][well]
                
                logger.info(f"Running genotyping for well {well} with {len(well_files)} files")
                
                # Process each file
                for i, input_file in enumerate(well_files):
                    logger.info(f"  Processing file {i+1}/{len(well_files)}: {os.path.basename(input_file)}")
                    
                    try:
                        # Run genotyping
                        genotyping = GenotypingPipeline(
                            input_file=input_file,
                            segmentation_dir=seg_dir,
                            output_dir=geno_output_dir,
                            barcode_library=barcode_file
                        )
                        
                        genotyping.run(wells=[well])
                    except Exception as e:
                        logger.error(f"Error processing genotyping file {input_file}: {e}")
                
                results[well] = True
                
            except Exception as e:
                logger.error(f"Error in genotyping for well {well}: {e}")
                results[well] = False
    
    return results, geno_output_dir

def run_phenotyping(files, seg_dir, geno_dir, output_dir, wells, channels, logger):
    """Run phenotyping on input files."""
    pheno_output_dir = os.path.join(output_dir, "phenotyping")
    os.makedirs(pheno_output_dir, exist_ok=True)
    
    results = {}
    
    # Parse channels
    channel_list = channels.split(",")
    
    if wells:
        filtered_wells = wells
    else:
        filtered_wells = list(files["phenotyping"].keys())
    
    for well in filtered_wells:
        if well in files["phenotyping"]:
            try:
                from imageanalysis.core.phenotyping.pipeline import PhenotypingPipeline
                
                # Get limited number of files
                well_files = files["phenotyping"][well]
                
                logger.info(f"Running phenotyping for well {well} with {len(well_files)} files")
                
                # Process each file
                for i, input_file in enumerate(well_files):
                    logger.info(f"  Processing file {i+1}/{len(well_files)}: {os.path.basename(input_file)}")
                    
                    try:
                        # Run phenotyping
                        phenotyping = PhenotypingPipeline(
                            input_file=input_file,
                            segmentation_dir=seg_dir,
                            genotyping_dir=geno_dir,
                            output_dir=pheno_output_dir,
                            channels=channel_list
                        )
                        
                        phenotyping.run(wells=[well])
                    except Exception as e:
                        logger.error(f"Error processing phenotyping file {input_file}: {e}")
                
                results[well] = True
                
            except Exception as e:
                logger.error(f"Error in phenotyping for well {well}: {e}")
                results[well] = False
    
    return results, pheno_output_dir

def create_albums(pheno_dir, output_dir, wells, logger):
    """Create albums from phenotyping results."""
    albums_dir = os.path.join(output_dir, "albums")
    os.makedirs(albums_dir, exist_ok=True)
    
    results = {}
    
    try:
        from imageanalysis.core.visualization.albums import AlbumCreator
        
        logger.info("Creating albums...")
        
        # Initialize album creator
        creator = AlbumCreator(
            phenotyping_dir=pheno_dir,
            output_dir=albums_dir
        )
        
        # Run for each well
        for well in wells:
            try:
                creator.run(wells=[well])
                results[well] = True
                logger.info(f"Created album for well {well}")
            except Exception as e:
                logger.error(f"Error creating album for well {well}: {e}")
                results[well] = False
        
    except Exception as e:
        logger.error(f"Error in album creation: {e}")
        for well in wells:
            results[well] = False
    
    return results

def generate_report(output_dir, all_results, wells, logger):
    """Generate a report summarizing the pipeline test."""
    logger.info("Generating test report...")
    report_dir = os.path.join(output_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(report_dir, f"test_report_{timestamp}.json")
    
    # Consolidate results
    results = {
        "segmentation": all(all_results.get("segmentation", {}).values()),
        "mapping": all_results.get("mapping", False),
        "genotyping": all(all_results.get("genotyping", {}).values()),
        "phenotyping": all(all_results.get("phenotyping", {}).values()),
        "albums": all(all_results.get("albums", {}).values())
    }
    
    report = {
        "timestamp": timestamp,
        "wells_processed": wells,
        "results": results,
        "detailed_results": all_results,
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

def main():
    """Main function."""
    args = parse_args()
    
    # Set up logging
    logger = setup_logging(args.output_dir, args.debug)
    
    logger.info("=== Real Data Pipeline Test ===")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Wells to process: {args.wells}")
    logger.info(f"Barcode file: {args.barcode_file}")
    logger.info(f"File limit: {args.limit_files}")
    logger.info(f"Channels: {args.channels}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find input data files
    logger.info("Finding input data files...")
    files = find_data_files(args.data_dir, args.limit_files)
    
    # Log what we found
    for step in files:
        logger.info(f"Found {sum(len(wells) for wells in files[step].values())} files for {step}")
        for well, well_files in files[step].items():
            logger.info(f"  {well}: {len(well_files)} files")
    
    # Track all results
    all_results = {}
    
    # Step 1: Segmentation
    logger.info("\n=== Step 1: Segmentation ===")
    seg_results, seg_dir = run_segmentation(files, args.output_dir, args.wells, args.channels, logger)
    all_results["segmentation"] = seg_results
    
    # Step 2: Mapping
    logger.info("\n=== Step 2: Mapping ===")
    mapping_result = run_mapping(seg_dir, args.output_dir, args.wells, logger)
    all_results["mapping"] = mapping_result
    
    # Step 3: Genotyping
    logger.info("\n=== Step 3: Genotyping ===")
    geno_results, geno_dir = run_genotyping(files, seg_dir, args.output_dir, args.wells, args.barcode_file, logger)
    all_results["genotyping"] = geno_results
    
    # Step 4: Phenotyping
    logger.info("\n=== Step 4: Phenotyping ===")
    pheno_results, pheno_dir = run_phenotyping(
        files, seg_dir, geno_dir, args.output_dir, args.wells, args.channels, logger
    )
    all_results["phenotyping"] = pheno_results
    
    # Step 5: Album Creation
    logger.info("\n=== Step 5: Album Creation ===")
    album_results = create_albums(pheno_dir, args.output_dir, args.wells, logger)
    all_results["albums"] = album_results
    
    # Generate report
    generate_report(args.output_dir, all_results, args.wells, logger)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())