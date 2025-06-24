#!/usr/bin/env python3
"""
Benchmark script to compare L-BFGS-B and Levenberg-Marquardt algorithms
using real data from the image analysis pipeline.

This script:
1. Processes 10x and 40x images through the segmentation pipeline
2. Extracts nuclear centroids from both magnifications
3. Evaluates both L-BFGS-B and LMA optimization algorithms on point matching
4. Measures RMSD, time, and memory performance

Results are saved to CSV files for analysis.
"""

import os
import time
import tracemalloc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
from pathlib import Path
import argparse
import logging
import json
import tempfile
import shutil
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# Import image analysis components
from imageanalysis.core.segmentation import Segmentation10XPipeline, Segmentation40XPipeline
from imageanalysis.core.mapping.matching import find_nearest_neighbors, ransac_filter
from imageanalysis.utils.logging import setup_logger

# Configure logging
logger = setup_logger('benchmark', level=logging.INFO)

def extract_nuclear_centroids(segmentation_dir, well_id):
    """
    Extract nuclear centroids from segmentation results.
    
    Args:
        segmentation_dir: Directory containing segmentation results
        well_id: Well identifier
        
    Returns:
        nuclear_centroids: Dictionary of file_id -> centroids array
    """
    logger.info(f"Extracting nuclear centroids from {segmentation_dir}/{well_id}")
    well_dir = Path(segmentation_dir) / well_id
    
    if not well_dir.exists():
        logger.error(f"Well directory not found: {well_dir}")
        return {}
    
    nuclear_centroids = {}
    centroid_files = list(well_dir.glob('*_nuclei_centroids_local.npy'))
    
    if not centroid_files:
        logger.warning(f"No nuclear centroid files found in {well_dir}")
        return {}
    
    for file_path in centroid_files:
        file_id = file_path.stem.replace('_nuclei_centroids_local', '')
        try:
            centroids = np.load(file_path)
            nuclear_centroids[file_id] = centroids
            logger.info(f"Loaded {len(centroids)} centroids from {file_id}")
        except Exception as e:
            logger.error(f"Error loading centroids from {file_path}: {e}")
    
    return nuclear_centroids

def perform_segmentation(image_file, output_dir, is_10x=True):
    """
    Perform segmentation on the specified image file.
    
    Args:
        image_file: Path to the image file
        output_dir: Output directory for segmentation results
        is_10x: Whether the image is 10x (True) or 40x (False)
        
    Returns:
        result: Segmentation result dictionary
    """
    # Create configuration
    config = {
        'input_file': str(image_file),
        'output_dir': str(output_dir),
        'nuclear_channel': 0,  # DAPI
        'cell_channel': 1      # Cell body
    }
    
    # Initialize and run the appropriate pipeline
    if is_10x:
        pipeline = Segmentation10XPipeline(config)
    else:
        pipeline = Segmentation40XPipeline(config)
    
    try:
        result = pipeline.run()
        return result
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        return {'status': 'error', 'error': str(e)}

def find_corresponding_points(centroids_10x, centroids_40x, max_num_points=500):
    """
    Find corresponding points between 10x and 40x centroids using RANSAC.
    This function emulates what would happen in the real mapping process
    but is simplified for benchmarking purposes.
    
    Args:
        centroids_10x: Dictionary of 10x centroids (file_id -> centroids)
        centroids_40x: Dictionary of 40x centroids (file_id -> centroids)
        max_num_points: Maximum number of points to use
        
    Returns:
        matched_10x: Array of matched 10x points
        matched_40x: Array of matched 40x points
    """
    # For simplicity, just take the first file from each magnification
    # In a real scenario, this would involve more sophisticated point matching
    file_10x = next(iter(centroids_10x.keys()))
    file_40x = next(iter(centroids_40x.keys()))
    
    points_10x = centroids_10x[file_10x]
    points_40x = centroids_40x[file_40x]
    
    # Limit number of points to avoid excessive computation
    if len(points_10x) > max_num_points:
        points_10x = points_10x[:max_num_points]
    if len(points_40x) > max_num_points:
        points_40x = points_40x[:max_num_points]
    
    # Use RANSAC to find initial correspondences
    # This is a simplified version of the actual point matching process
    # In a real mapping scenario, additional constraints would be applied
    try:
        # Get approximate scale between 10x and 40x (typically around 4x)
        scale = 4.0
        
        # Center the point sets (removes translation component)
        center_10x = np.mean(points_10x, axis=0)
        center_40x = np.mean(points_40x, axis=0)
        centered_10x = points_10x - center_10x
        centered_40x = points_40x - center_40x
        
        # Scale the 10x points to approximate 40x scale
        scaled_10x = centered_10x * scale
        
        # Use find_nearest_neighbors followed by ransac_filter
        # This is similar to what the mapping pipeline would do
        try:
            # Find initial correspondences with nearest neighbors
            matched_scaled_10x, matched_40x = find_nearest_neighbors(
                scaled_10x, centered_40x,
                max_distance=100.0
            )

            # Apply RANSAC filtering to remove outliers
            matched_scaled_10x, matched_40x = ransac_filter(
                matched_scaled_10x, matched_40x,
                num_iterations=1000,
                inlier_threshold=20.0
            )

            # Map back to original points
            indices_10x = []
            indices_40x = []

            # Find the original indices of the matched points
            for i, p1 in enumerate(matched_scaled_10x):
                # Find matching point in original scaled_10x
                idx1 = np.argmin(np.sum((scaled_10x - p1)**2, axis=1))
                # Find matching point in original centered_40x
                p2 = matched_40x[i]
                idx2 = np.argmin(np.sum((centered_40x - p2)**2, axis=1))

                indices_10x.append(idx1)
                indices_40x.append(idx2)

            success = True
        except Exception as e:
            logger.warning(f"RANSAC filtering failed: {e}")
            success = False

        if not success or len(indices_10x) < 10:
            logger.warning(f"Failed to find enough correspondences: {len(indices_10x) if success else 0}")
            # Take random subsets as fallback
            min_size = min(len(points_10x), len(points_40x), 100)
            indices_10x = np.random.choice(len(points_10x), min_size, replace=False)
            indices_40x = np.random.choice(len(points_40x), min_size, replace=False)
        
        # Get the corresponding points
        matched_10x = points_10x[indices_10x]
        matched_40x = points_40x[indices_40x]
        
        logger.info(f"Found {len(matched_10x)} corresponding points between 10x and 40x")
        return matched_10x, matched_40x
        
    except Exception as e:
        logger.error(f"Error finding corresponding points: {e}")
        # Return subsets as fallback
        min_size = min(len(points_10x), len(points_40x), 50)
        return points_10x[:min_size], points_40x[:min_size]

def transform_points(points, params):
    """
    Apply affine transformation to points.
    
    Args:
        points: Array of points (Nx2)
        params: Transformation parameters [a, b, tx, c, d, ty]
        
    Returns:
        transformed_points: Transformed points (Nx2)
    """
    a, b, tx, c, d, ty = params
    transformed = np.zeros_like(points)
    transformed[:, 0] = a * points[:, 0] + b * points[:, 1] + tx
    transformed[:, 1] = c * points[:, 0] + d * points[:, 1] + ty
    return transformed

def calculate_rmsd(src_points, dst_points, params):
    """
    Calculate Root Mean Square Deviation between transformed source points and destination points.
    
    Args:
        src_points: Source points (Nx2)
        dst_points: Destination points (Nx2)
        params: Transformation parameters
        
    Returns:
        rmsd: Root Mean Square Deviation
    """
    transformed = transform_points(src_points, params)
    squared_diffs = np.sum((transformed - dst_points) ** 2, axis=1)
    return np.sqrt(np.mean(squared_diffs))

def lbfgsb_objective(params, src_points, dst_points):
    """
    Objective function for L-BFGS-B optimizer.
    
    Args:
        params: Transformation parameters
        src_points: Source points
        dst_points: Destination points
        
    Returns:
        error: Sum of squared distances between transformed source and destination points
    """
    transformed = transform_points(src_points, params)
    return np.sum((transformed - dst_points) ** 2)

def lma_objective(params, src_points, dst_points):
    """
    Objective function for Levenberg-Marquardt optimizer.
    
    Args:
        params: Transformation parameters
        src_points: Source points
        dst_points: Destination points
        
    Returns:
        residuals: Flattened array of residuals (x and y differences)
    """
    transformed = transform_points(src_points, params)
    residuals = (transformed - dst_points).flatten()
    return residuals

def get_initial_params():
    """
    Get initial transformation parameters for 10x to 40x mapping.
    For real data, we'd use known approximate magnification difference.
    
    Returns:
        params: Initial transformation parameters
    """
    # Start with approximate scale difference (10x to 40x = ~4x)
    # [a, b, tx, c, d, ty] where a,d=scale, b,c=rotation/shear, tx,ty=translation
    return np.array([4.0, 0.0, 0.0, 0.0, 4.0, 0.0])

def optimize_lbfgsb(src_points, dst_points, initial_params, max_iter=100):
    """
    Optimize transformation using L-BFGS-B algorithm.
    
    Args:
        src_points: Source points (10x)
        dst_points: Destination points (40x)
        initial_params: Initial transformation parameters
        max_iter: Maximum number of iterations
        
    Returns:
        params: Optimized parameters
        time_taken: Computation time
        max_memory: Peak memory usage
        iterations: Number of iterations
        rmsd: Root Mean Square Deviation after optimization
    """
    tracemalloc.start()
    start_time = time.time()
    
    result = minimize(
        lbfgsb_objective,
        initial_params,
        args=(src_points, dst_points),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'disp': False}
    )
    
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    time_taken = end_time - start_time
    max_memory = peak / 1024  # Convert to KB
    rmsd = calculate_rmsd(src_points, dst_points, result.x)
    
    return result.x, time_taken, max_memory, result.nit, rmsd

def optimize_lma(src_points, dst_points, initial_params, max_iter=100):
    """
    Optimize transformation using Levenberg-Marquardt algorithm.
    
    Args:
        src_points: Source points (10x)
        dst_points: Destination points (40x)
        initial_params: Initial transformation parameters
        max_iter: Maximum number of iterations
        
    Returns:
        params: Optimized parameters
        time_taken: Computation time
        max_memory: Peak memory usage
        iterations: Number of iterations
        rmsd: Root Mean Square Deviation after optimization
    """
    tracemalloc.start()
    start_time = time.time()
    
    result = least_squares(
        lma_objective,
        initial_params,
        args=(src_points, dst_points),
        method='lm',
        max_nfev=max_iter
    )
    
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    time_taken = end_time - start_time
    max_memory = peak / 1024  # Convert to KB
    rmsd = calculate_rmsd(src_points, dst_points, result.x)
    
    return result.x, time_taken, max_memory, result.nfev, rmsd

def run_benchmark_on_real_data(data_dir, output_dir, num_trials=3):
    """
    Run benchmark comparing L-BFGS-B and LMA algorithms on real data.
    
    Args:
        data_dir: Directory containing image data
        output_dir: Directory to save results
        num_trials: Number of trials to run
        
    Returns:
        results_df: DataFrame with benchmark results
    """
    logger.info("Starting real data benchmark...")
    data_dir = Path(data_dir)
    results_dir = Path(output_dir)
    
    # Create output directories
    os.makedirs(results_dir, exist_ok=True)

    # Create a unique temporary directory that won't conflict with parallel runs
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pid = os.getpid()
    temp_dir = Path(tempfile.mkdtemp(prefix=f"optimizer_benchmark_{timestamp}_{pid}_"))
    
    try:
        # Create subdirectories for segmentation results
        seg_10x_dir = temp_dir / "segmentation_10x"
        seg_40x_dir = temp_dir / "segmentation_40x"
        os.makedirs(seg_10x_dir, exist_ok=True)
        os.makedirs(seg_40x_dir, exist_ok=True)
        
        # Find 10x and 40x image files
        image_10x_dir = data_dir / "genotyping" / "cycle_1"
        image_40x_dir = data_dir / "phenotyping"
        
        image_10x_files = list(image_10x_dir.glob("*.nd2"))
        image_40x_files = list(image_40x_dir.glob("*.nd2"))
        
        if not image_10x_files or not image_40x_files:
            logger.error(f"No image files found: 10x={len(image_10x_files)}, 40x={len(image_40x_files)}")
            return pd.DataFrame()
        
        # Limit to a few files for benchmarking
        image_10x_files = image_10x_files[:2]
        image_40x_files = image_40x_files[:2]
        
        logger.info(f"Found {len(image_10x_files)} 10x images and {len(image_40x_files)} 40x images")
        
        # Perform segmentation on 10x and 40x images in parallel
        # Determine number of cores to use (leave one core free)
        num_cores = max(1, multiprocessing.cpu_count() - 1)
        logger.info(f"Using {num_cores} CPU cores for parallel segmentation")

        # Create parallel processing functions for 10x and 40x segmentation
        seg_10x_partial = partial(perform_segmentation, output_dir=seg_10x_dir, is_10x=True)
        seg_40x_partial = partial(perform_segmentation, output_dir=seg_40x_dir, is_10x=False)

        # Process 10x and 40x images in parallel with ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Queue both 10x and 40x images for processing
            logger.info(f"Submitting segmentation tasks for {len(image_10x_files)} 10x and {len(image_40x_files)} 40x images")

            # Submit both 10x and 40x tasks in parallel to maximize core usage
            futures_10x = [executor.submit(seg_10x_partial, img_file) for img_file in image_10x_files]
            futures_40x = [executor.submit(seg_40x_partial, img_file) for img_file in image_40x_files]

            # Wait for all tasks to complete
            all_futures = futures_10x + futures_40x

            # Process results as they complete rather than waiting for all to finish
            results_10x = []
            results_40x = []

            for i, future in enumerate(futures_10x):
                result = future.result()
                results_10x.append(result)
                logger.info(f"Completed 10x segmentation {i+1}/{len(futures_10x)}: {image_10x_files[i].name}")

            for i, future in enumerate(futures_40x):
                result = future.result()
                results_40x.append(result)
                logger.info(f"Completed 40x segmentation {i+1}/{len(futures_40x)}: {image_40x_files[i].name}")

            logger.info(f"All segmentation tasks completed. 10x results: {len(results_10x)}, 40x results: {len(results_40x)}")
        
        # Extract nuclear centroids
        centroids_10x = extract_nuclear_centroids(seg_10x_dir, "Well1")
        centroids_40x = extract_nuclear_centroids(seg_40x_dir, "Well1")
        
        if not centroids_10x or not centroids_40x:
            logger.error("Failed to extract enough centroids for comparison")
            return pd.DataFrame()
        
        # Find corresponding points
        logger.info("Finding corresponding points between 10x and 40x")
        points_10x, points_40x = find_corresponding_points(centroids_10x, centroids_40x)
        
        if len(points_10x) < 10 or len(points_40x) < 10:
            logger.error(f"Not enough corresponding points: {len(points_10x)}, {len(points_40x)}")
            return pd.DataFrame()
        
        # Run benchmarks in parallel across trials
        results = []

        def run_optimizer_trial(trial_num):
            """Run a single trial of both optimizers in parallel"""
            trial_results = []
            logger.info(f"Running trial {trial_num+1}/{num_trials}")

            # Use same initial parameters for both optimizers
            initial_params = get_initial_params()

            # Run L-BFGS-B
            lbfgsb_params, lbfgsb_time, lbfgsb_memory, lbfgsb_iter, lbfgsb_rmsd = optimize_lbfgsb(
                points_10x, points_40x, initial_params.copy()
            )

            # Run LMA
            lma_params, lma_time, lma_memory, lma_iter, lma_rmsd = optimize_lma(
                points_10x, points_40x, initial_params.copy()
            )

            # Store results
            trial_results.append({
                'trial': trial_num,
                'num_points': len(points_10x),
                'optimization': 'L-BFGS-B',
                'rmsd': lbfgsb_rmsd,
                'time_ms': lbfgsb_time * 1000,  # Convert to milliseconds
                'memory_kb': lbfgsb_memory,
                'iterations': lbfgsb_iter,
                'a': lbfgsb_params[0],
                'b': lbfgsb_params[1],
                'tx': lbfgsb_params[2],
                'c': lbfgsb_params[3],
                'd': lbfgsb_params[4],
                'ty': lbfgsb_params[5]
            })

            trial_results.append({
                'trial': trial_num,
                'num_points': len(points_10x),
                'optimization': 'LMA',
                'rmsd': lma_rmsd,
                'time_ms': lma_time * 1000,  # Convert to milliseconds
                'memory_kb': lma_memory,
                'iterations': lma_iter,
                'a': lma_params[0],
                'b': lma_params[1],
                'tx': lma_params[2],
                'c': lma_params[3],
                'd': lma_params[4],
                'ty': lma_params[5]
            })

            return trial_results

        # Run trials in parallel
        logger.info(f"Running {num_trials} benchmark trials in parallel")
        with ProcessPoolExecutor(max_workers=min(num_cores, num_trials)) as executor:
            # Submit all trials for parallel execution
            futures = [executor.submit(run_optimizer_trial, trial) for trial in range(num_trials)]

            # Collect results as they complete
            for future in futures:
                trial_results = future.result()
                results.extend(trial_results)
                logger.info(f"Completed trial {len(results)//2}/{num_trials}")
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results to CSV
        csv_path = results_dir / 'optimizer_benchmark_real_data.csv'
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")
        
        # Create summary
        summary = results_df.groupby('optimization').agg({
            'rmsd': ['mean', 'std'],
            'time_ms': ['mean', 'std'],
            'memory_kb': ['mean', 'std'],
            'iterations': ['mean', 'std']
        })
        
        summary_path = results_dir / 'optimizer_benchmark_real_data_summary.csv'
        summary.to_csv(summary_path)
        logger.info(f"Summary saved to {summary_path}")
        
        # Plot results
        plot_dir = results_dir / 'plots'
        os.makedirs(plot_dir, exist_ok=True)
        
        # Plot RMSD comparison
        plt.figure(figsize=(10, 6))
        for opt in ['L-BFGS-B', 'LMA']:
            subset = results_df[results_df['optimization'] == opt]
            plt.scatter(subset['trial'], subset['rmsd'], label=opt, alpha=0.7)
        plt.xlabel('Trial')
        plt.ylabel('RMSD')
        plt.title('RMSD Comparison on Real Data')
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_dir / 'rmsd_comparison_real_data.png')
        
        # Plot time comparison
        plt.figure(figsize=(10, 6))
        for opt in ['L-BFGS-B', 'LMA']:
            subset = results_df[results_df['optimization'] == opt]
            plt.scatter(subset['trial'], subset['time_ms'], label=opt, alpha=0.7)
        plt.xlabel('Trial')
        plt.ylabel('Time (ms)')
        plt.title('Computation Time Comparison on Real Data')
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_dir / 'time_comparison_real_data.png')
        
        return results_df
        
    except Exception as e:
        logger.error(f"Error in benchmark: {e}")
        return pd.DataFrame()
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description='Benchmark L-BFGS-B vs LMA algorithms on real data')
    parser.add_argument('--data-dir', type=str, default='/Users/csm70/Desktop/Shalem_Lab/ImageAnalysis/Data',
                        help='Directory containing image data')
    parser.add_argument('--output-dir', type=str, default='results/optimizer_benchmark_real_data',
                        help='Directory to save results')
    parser.add_argument('--trials', type=int, default=3, help='Number of trials to run')
    args = parser.parse_args()

    # Track total execution time
    total_start_time = time.time()

    # Display CPU information
    num_cores = multiprocessing.cpu_count()
    logger.info(f"Running on system with {num_cores} CPU cores")

    # Run benchmark on real data
    results_df = run_benchmark_on_real_data(args.data_dir, args.output_dir, args.trials)

    # Calculate total execution time
    total_time = time.time() - total_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info(f"Real data benchmark completed successfully!")
    logger.info(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
if __name__ == "__main__":
    main()