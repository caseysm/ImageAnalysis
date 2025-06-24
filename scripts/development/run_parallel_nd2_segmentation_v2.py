#!/usr/bin/env python3
"""
Script to run segmentation on multiple ND2 files in parallel,
using the ND2Segmentation class from nd2_segmentation_v2.py.
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import signal

from nd2_segmentation_v2 import run_nd2_segmentation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('parallel-nd2-segmentation')

def find_nd2_files(input_dir, pattern="*.nd2"):
    """Find all ND2 files in the input directory.
    
    Args:
        input_dir: Directory to search for ND2 files
        pattern: Pattern to match ND2 files
        
    Returns:
        list: List of ND2 file paths
    """
    input_path = Path(input_dir)
    return list(input_path.glob(f"**/{pattern}"))

def get_magnification_from_path(file_path):
    """Determine the magnification from the file path.
    
    Args:
        file_path: Path to the ND2 file
        
    Returns:
        str: '10X' or '40X'
    """
    path_str = str(file_path).lower()
    
    # Try to determine from directory structure
    if "genotyping" in path_str or "10x" in path_str:
        return "10X"
    elif "phenotyping" in path_str or "40x" in path_str:
        return "40X"
    
    # Default to 10X
    return "10X"

def init_worker():
    """Initialize worker process to make keyboard interrupts work."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def process_file(nd2_file, output_dir, progress_file=None):
    """Process a single ND2 file.
    
    Args:
        nd2_file: Path to ND2 file
        output_dir: Directory to save results
        progress_file: Path to progress file
        
    Returns:
        dict: Results of segmentation
    """
    try:
        file_path = Path(nd2_file)
        file_id = file_path.stem
        
        # Determine magnification
        magnification = get_magnification_from_path(file_path)
        
        # Create full output directory
        full_output_dir = os.path.join(output_dir)
        
        logger.info(f"Processing {file_id} at {magnification} magnification")
        
        # Run segmentation
        results = run_nd2_segmentation(
            nd2_file=str(file_path),
            output_dir=full_output_dir,
            magnification=magnification,
            nuclear_channel=0,  # DAPI usually channel 0
            cell_channel=1      # Cell marker usually channel 1
        )
        
        # Update progress if needed
        if progress_file:
            with open(progress_file, 'a') as f:
                f.write(f"{file_id},{results['status']}\n")
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing {nd2_file}: {e}")
        
        # Update progress if needed
        if progress_file:
            with open(progress_file, 'a') as f:
                f.write(f"{file_id},error: {str(e)}\n")
        
        return {
            "status": "error",
            "file": str(file_path),
            "error": str(e)
        }

def run_parallel_segmentation(input_dir, output_dir, num_cores, limit=None):
    """Run segmentation on multiple ND2 files in parallel.
    
    Args:
        input_dir: Directory containing ND2 files
        output_dir: Directory to save segmentation results
        num_cores: Number of CPU cores to use
        limit: Maximum number of files to process
        
    Returns:
        dict: Summary of results
    """
    start_time = time.time()
    
    # Find all ND2 files
    nd2_files = find_nd2_files(input_dir)
    
    if limit and limit > 0:
        nd2_files = nd2_files[:limit]
    
    num_files = len(nd2_files)
    logger.info(f"Found {num_files} ND2 files")
    
    if num_files == 0:
        logger.error(f"No ND2 files found in {input_dir}")
        return {"status": "error", "message": "No ND2 files found"}
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create progress file
    progress_file = os.path.join(output_dir, "segmentation_progress.csv")
    with open(progress_file, 'w') as f:
        f.write("file,status\n")
    
    # Create a partial function with fixed arguments
    process_fn = partial(process_file, output_dir=output_dir, progress_file=progress_file)
    
    # Process files in parallel
    results = []
    
    try:
        with ProcessPoolExecutor(max_workers=num_cores, initializer=init_worker) as executor:
            futures = {executor.submit(process_fn, nd2_file): nd2_file for nd2_file in nd2_files}
            
            # Show progress bar
            with tqdm(total=num_files, desc="Segmenting ND2 files") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
    
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt detected. Cancelling tasks...")
        for future in futures:
            future.cancel()
        logger.warning("Tasks cancelled. Exiting gracefully.")
        sys.exit(1)
    
    # Calculate total processing time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Create summary
    summary = {
        "total_files": num_files,
        "successful": len([r for r in results if r.get("status") == "success"]),
        "failed": len([r for r in results if r.get("status") != "success"]),
        "processing_time": f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    }
    
    # Save summary
    summary_path = os.path.join(output_dir, "segmentation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Segmentation completed: {summary['successful']}/{num_files} successful")
    logger.info(f"Total processing time: {summary['processing_time']}")
    
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run parallel segmentation on multiple ND2 files')
    parser.add_argument('--input', required=True, help='Directory containing ND2 files')
    parser.add_argument('--output', required=True, help='Directory to save segmentation results')
    parser.add_argument('--cores', type=int, default=None, help='Number of CPU cores to use. Default is all available cores - 1')
    parser.add_argument('--limit', type=int, default=None, help='Maximum number of files to process')
    
    args = parser.parse_args()
    
    # Determine the number of cores to use
    if args.cores is None:
        import multiprocessing
        args.cores = max(1, multiprocessing.cpu_count() - 1)
    
    run_parallel_segmentation(args.input, args.output, args.cores, args.limit)