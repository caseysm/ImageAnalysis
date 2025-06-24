#!/usr/bin/env python3
"""
Benchmark script to compare L-BFGS-B and Levenberg-Marquardt algorithms
using nuclear centroids from segmentation results.

This script:
1. Loads nuclear centroids from segmentation output
2. Pairs corresponding 10x and 40x images
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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('benchmark_optimizers')

def extract_centroids_from_segmentation(segmentation_dir):
    """
    Extract nuclear centroids from segmentation results.
    
    Args:
        segmentation_dir: Directory containing segmentation results
        
    Returns:
        centroids_10x: Dictionary of well_id -> file_id -> centroids for 10x images
        centroids_40x: Dictionary of well_id -> file_id -> centroids for 40x images
    """
    segmentation_dir = Path(segmentation_dir)
    logger.info(f"Extracting centroids from {segmentation_dir}")
    
    # Directories for 10x and 40x results
    dir_10x = segmentation_dir / "10X"
    dir_40x = segmentation_dir / "40X"

    # Log which directories exist
    logger.info(f"Checking segmentation directories: 10X={dir_10x.exists()}, 40X={dir_40x.exists()}")

    # Only require at least one directory to exist
    if not dir_10x.exists() and not dir_40x.exists():
        logger.error(f"No segmentation directories found: 10X={dir_10x.exists()}, 40X={dir_40x.exists()}")
        return {}, {}
    
    # Extract centroids from 10x results
    centroids_10x = {}
    if dir_10x.exists():
        for well_dir in dir_10x.iterdir():
            if well_dir.is_dir():
                well_id = well_dir.name
                centroids_10x[well_id] = {}

                centroid_files = list(well_dir.glob('*_nuclei_centroids_local.npy'))
                for file_path in centroid_files:
                    file_id = file_path.stem.replace('_nuclei_centroids_local', '')
                    try:
                        centroids = np.load(file_path)
                        centroids_10x[well_id][file_id] = centroids
                        logger.info(f"Loaded {len(centroids)} 10X centroids from {well_id}/{file_id}")
                    except Exception as e:
                        logger.error(f"Error loading 10X centroids from {file_path}: {e}")
    else:
        logger.info("No 10X directory found, skipping 10X centroids")

    # Extract centroids from 40x results
    centroids_40x = {}
    if dir_40x.exists():
        for well_dir in dir_40x.iterdir():
            if well_dir.is_dir():
                well_id = well_dir.name
                centroids_40x[well_id] = {}

                centroid_files = list(well_dir.glob('*_nuclei_centroids_local.npy'))
                for file_path in centroid_files:
                    file_id = file_path.stem.replace('_nuclei_centroids_local', '')
                    try:
                        centroids = np.load(file_path)
                        centroids_40x[well_id][file_id] = centroids
                        logger.info(f"Loaded {len(centroids)} 40X centroids from {well_id}/{file_id}")
                    except Exception as e:
                        logger.error(f"Error loading 40X centroids from {file_path}: {e}")
    else:
        logger.info("No 40X directory found, skipping 40X centroids")
    
    logger.info(f"Extracted centroids from {len(centroids_10x)} 10X wells and {len(centroids_40x)} 40X wells")
    return centroids_10x, centroids_40x

def find_corresponding_points(centroids_10x, centroids_40x, max_num_points=500):
    """
    Find corresponding points between 10x and 40x centroids.
    
    Args:
        centroids_10x: Dictionary of well_id -> file_id -> centroids for 10x images
        centroids_40x: Dictionary of well_id -> file_id -> centroids for 40x images
        max_num_points: Maximum number of points to use
        
    Returns:
        point_sets: List of dictionaries with matched 10x and 40x points
    """
    point_sets = []
    
    # For each well_id in 10x data
    for well_id, files_10x in centroids_10x.items():
        if well_id not in centroids_40x:
            logger.warning(f"No matching 40X data for 10X well {well_id}")
            continue
        
        files_40x = centroids_40x[well_id]
        
        # For each file_id in the well
        for file_id_10x, points_10x in files_10x.items():
            # Try to find a matching 40x file
            # In a real pipeline, you would have a more sophisticated matching method
            # Here we'll just use a simple heuristic: take the first available 40x file
            if not files_40x:
                logger.warning(f"No 40X files for well {well_id}")
                continue
            
            # Take the first 40x file for now
            file_id_40x = next(iter(files_40x.keys()))
            points_40x = files_40x[file_id_40x]
            
            # Limit number of points
            if len(points_10x) > max_num_points:
                points_10x = points_10x[:max_num_points]
            if len(points_40x) > max_num_points:
                points_40x = points_40x[:max_num_points]
            
            # Find matching points (simplified version)
            try:
                # Get approximate scale between 10x and 40x (typically around 4x)
                scale = 4.0
                
                # Center the point sets
                center_10x = np.mean(points_10x, axis=0)
                center_40x = np.mean(points_40x, axis=0)
                centered_10x = points_10x - center_10x
                centered_40x = points_40x - center_40x
                
                # Scale the 10x points to approximate 40x scale
                scaled_10x = centered_10x * scale
                
                # In the real pipeline, you would use find_nearest_neighbors and ransac_filter
                # For this benchmark, we'll use a simpler approach: just take a random subset
                min_size = min(len(points_10x), len(points_40x), 100)
                indices_10x = np.random.choice(len(points_10x), min_size, replace=False)
                indices_40x = np.random.choice(len(points_40x), min_size, replace=False)
                
                # Get the corresponding points
                matched_10x = points_10x[indices_10x]
                matched_40x = points_40x[indices_40x]
                
                # Store the point set
                point_sets.append({
                    'well_id': well_id,
                    'file_10x': file_id_10x,
                    'file_40x': file_id_40x,
                    'points_10x': matched_10x,
                    'points_40x': matched_40x,
                    'num_points': len(matched_10x)
                })
                
                logger.info(f"Matched {len(matched_10x)} points between 10X:{file_id_10x} and 40X:{file_id_40x} in well {well_id}")
                
            except Exception as e:
                logger.error(f"Error finding corresponding points for well {well_id}: {e}")
    
    logger.info(f"Found {len(point_sets)} matched point sets across all wells")
    return point_sets

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

def run_optimizer_trial(point_set, trial_num, num_iterations):
    """
    Run a single optimizer trial on a given point set.
    
    Args:
        point_set: Dictionary with matched 10x and 40x points
        trial_num: Trial number
        num_iterations: Maximum number of iterations
        
    Returns:
        results: List of result dictionaries
    """
    results = []
    
    well_id = point_set['well_id']
    file_10x = point_set['file_10x']
    file_40x = point_set['file_40x']
    points_10x = point_set['points_10x']
    points_40x = point_set['points_40x']
    num_points = point_set['num_points']
    
    logger.info(f"Running trial {trial_num} on well {well_id} with {num_points} points")
    
    # Get initial parameters
    initial_params = get_initial_params()
    
    # Run L-BFGS-B
    lbfgsb_params, lbfgsb_time, lbfgsb_memory, lbfgsb_iter, lbfgsb_rmsd = optimize_lbfgsb(
        points_10x, points_40x, initial_params.copy(), max_iter=num_iterations
    )
    
    # Run LMA
    lma_params, lma_time, lma_memory, lma_iter, lma_rmsd = optimize_lma(
        points_10x, points_40x, initial_params.copy(), max_iter=num_iterations
    )
    
    # Store results for L-BFGS-B
    results.append({
        'trial': trial_num,
        'well_id': well_id,
        'file_10x': file_10x,
        'file_40x': file_40x,
        'num_points': num_points,
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
    
    # Store results for LMA
    results.append({
        'trial': trial_num,
        'well_id': well_id,
        'file_10x': file_10x,
        'file_40x': file_40x,
        'num_points': num_points,
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
    
    logger.info(f"Completed trial {trial_num}: L-BFGS-B RMSD={lbfgsb_rmsd:.4f}, LMA RMSD={lma_rmsd:.4f}")
    return results

def run_benchmarks(point_sets, num_trials, num_iterations, output_dir):
    """
    Run benchmarks on all point sets.
    
    Args:
        point_sets: List of dictionaries with matched 10x and 40x points
        num_trials: Number of trials to run
        num_iterations: Maximum number of iterations
        output_dir: Directory to save results
        
    Returns:
        results_df: DataFrame with benchmark results
    """
    logger.info(f"Running benchmarks on {len(point_sets)} point sets with {num_trials} trials each")
    results_dir = Path(output_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create benchmark configurations
    benchmark_configs = []
    trial_id = 0
    
    for point_set in point_sets:
        for trial in range(num_trials):
            benchmark_configs.append((point_set, trial, num_iterations))
            trial_id += 1
    
    # Determine number of workers (leave one core free)
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    logger.info(f"Running {len(benchmark_configs)} benchmark configurations on {num_cores} CPU cores")
    
    # Run benchmarks in parallel
    results = []
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit all benchmark configurations
        futures = [executor.submit(run_optimizer_trial, *config) for config in benchmark_configs]
        
        # Collect results as they complete
        for i, future in enumerate(futures):
            try:
                trial_results = future.result()
                results.extend(trial_results)
                logger.info(f"Processed benchmark {i+1}/{len(futures)}")
            except Exception as e:
                logger.error(f"Error in benchmark {i}: {e}")
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    csv_path = results_dir / 'optimizer_benchmark_real_points.csv'
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")
    
    # Create summary by optimizer and well
    summary = results_df.groupby(['optimization', 'well_id']).agg({
        'rmsd': ['mean', 'std'],
        'time_ms': ['mean', 'std'],
        'memory_kb': ['mean', 'std'],
        'iterations': ['mean', 'std'],
        'num_points': ['mean']
    })
    
    summary_path = results_dir / 'optimizer_benchmark_summary.csv'
    summary.to_csv(summary_path)
    logger.info(f"Summary saved to {summary_path}")
    
    # Create overall summary
    overall = results_df.groupby(['optimization']).agg({
        'rmsd': ['mean', 'std', 'min', 'max'],
        'time_ms': ['mean', 'std', 'min', 'max'],
        'memory_kb': ['mean', 'std', 'min', 'max'],
        'iterations': ['mean', 'std'],
        'num_points': ['mean']
    })
    
    overall_path = results_dir / 'optimizer_benchmark_overall.csv'
    overall.to_csv(overall_path)
    logger.info(f"Overall summary saved to {overall_path}")
    
    # Plot results
    plot_dir = results_dir / 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot RMSD comparison
    plt.figure(figsize=(12, 8))
    for opt in ['L-BFGS-B', 'LMA']:
        subset = results_df[results_df['optimization'] == opt]
        plt.scatter(subset['num_points'], subset['rmsd'], label=opt, alpha=0.7)
    plt.xlabel('Number of Points')
    plt.ylabel('RMSD')
    plt.title('RMSD vs Number of Points')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_dir / 'rmsd_vs_points.png')
    
    # Plot time comparison
    plt.figure(figsize=(12, 8))
    for opt in ['L-BFGS-B', 'LMA']:
        subset = results_df[results_df['optimization'] == opt]
        plt.scatter(subset['num_points'], subset['time_ms'], label=opt, alpha=0.7)
    plt.xlabel('Number of Points')
    plt.ylabel('Time (ms)')
    plt.title('Computation Time vs Number of Points')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_dir / 'time_vs_points.png')
    
    # Plot RMSD by well
    plt.figure(figsize=(14, 8))
    well_ids = sorted(results_df['well_id'].unique())
    
    for opt in ['L-BFGS-B', 'LMA']:
        data = []
        for well_id in well_ids:
            subset = results_df[(results_df['optimization'] == opt) & (results_df['well_id'] == well_id)]
            data.append(subset['rmsd'].mean())
        
        plt.bar(
            [i + (0.4 if opt == 'LMA' else 0) for i in range(len(well_ids))],
            data,
            width=0.4,
            label=opt
        )
    
    plt.xlabel('Well ID')
    plt.ylabel('Mean RMSD')
    plt.title('Mean RMSD by Well ID')
    plt.xticks(range(len(well_ids)), well_ids, rotation=45)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(plot_dir / 'rmsd_by_well.png')
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Benchmark L-BFGS-B vs LMA algorithms on real nuclear centroids')
    parser.add_argument('--centroids-dir', type=str, default='results/segmentation_output', 
                        help='Directory containing segmentation results with nuclear centroids')
    parser.add_argument('--output-dir', type=str, default='results/optimizer_benchmark', 
                        help='Directory to save benchmark results')
    parser.add_argument('--trials', type=int, default=3, 
                        help='Number of trials to run for each point set')
    parser.add_argument('--iterations', type=int, default=100, 
                        help='Maximum optimizer iterations')
    args = parser.parse_args()
    
    # Track total execution time
    total_start_time = time.time()
    
    # Display CPU information
    cpu_count = multiprocessing.cpu_count()
    logger.info(f"Running on system with {cpu_count} CPU cores")
    
    # Extract centroids from segmentation results
    centroids_10x, centroids_40x = extract_centroids_from_segmentation(args.centroids_dir)
    
    # If we only have one set of centroids, generate a mock set for the other magnification
    if not centroids_10x and centroids_40x:
        logger.warning("Only 40X centroids found. Creating synthetic 10X centroids.")
        # Create synthetic 10X data from 40X data
        centroids_10x = {}
        for well_id, files in centroids_40x.items():
            centroids_10x[well_id] = {}
            for file_id, centroids in files.items():
                # Scale 40X to 10X (divide by 4) with some noise
                np.random.seed(0)  # For reproducibility
                scaled_centroids = centroids / 4.0
                # Add a little noise to make it interesting
                noise = np.random.normal(0, 5, scaled_centroids.shape)
                centroids_10x[well_id][file_id] = scaled_centroids + noise
                logger.info(f"Created {len(scaled_centroids)} synthetic 10X centroids for {well_id}/{file_id}")

    elif centroids_10x and not centroids_40x:
        logger.warning("Only 10X centroids found. Creating synthetic 40X centroids.")
        # Create synthetic 40X data from 10X data
        centroids_40x = {}
        for well_id, files in centroids_10x.items():
            centroids_40x[well_id] = {}
            for file_id, centroids in files.items():
                # Scale 10X to 40X (multiply by 4) with some noise
                np.random.seed(0)  # For reproducibility
                scaled_centroids = centroids * 4.0
                # Add a little noise to make it interesting
                noise = np.random.normal(0, 5, scaled_centroids.shape)
                centroids_40x[well_id][file_id] = scaled_centroids + noise
                logger.info(f"Created {len(scaled_centroids)} synthetic 40X centroids for {well_id}/{file_id}")

    elif not centroids_10x and not centroids_40x:
        logger.error("Failed to extract any centroids from segmentation results")
        return

    # Find corresponding points
    point_sets = find_corresponding_points(centroids_10x, centroids_40x)
    
    if not point_sets:
        logger.error("Failed to find corresponding points for optimization")
        return
    
    # Run benchmarks
    run_benchmarks(point_sets, args.trials, args.iterations, args.output_dir)
    
    # Calculate total execution time
    total_time = time.time() - total_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Optimizer benchmark completed successfully!")
    logger.info(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

if __name__ == "__main__":
    main()