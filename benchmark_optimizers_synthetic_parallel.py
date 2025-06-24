#!/usr/bin/env python3
"""
Parallel benchmark script to compare L-BFGS-B and Levenberg-Marquardt algorithms
using synthetic data, optimized for multi-core execution.

This script:
1. Generates synthetic nuclear centroids resembling real microscopy data
2. Evaluates both L-BFGS-B and LMA optimization algorithms on synthetic point matching
3. Measures RMSD, time, and memory performance
4. Utilizes all available CPU cores for parallel execution

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
logger = logging.getLogger('benchmark_synthetic')

def generate_nuclear_coordinates(num_points=1000, width=1024, height=1024, 
                               min_distance=10, max_retries=1000):
    """
    Generate synthetic nuclear centroids resembling real microscopy data.
    
    Args:
        num_points: Number of points to generate
        width: Image width in pixels
        height: Image height in pixels
        min_distance: Minimum distance between points
        max_retries: Maximum number of retries for each point
        
    Returns:
        points: Array of point coordinates (Nx2)
    """
    points = np.zeros((num_points, 2))
    current_points = 0
    
    # Start with some initial random points
    while current_points < num_points:
        # Generate a candidate point
        candidate = np.array([
            np.random.uniform(0, width),
            np.random.uniform(0, height)
        ])
        
        # Check if it's far enough from existing points
        if current_points > 0:
            distances = np.sqrt(np.sum((points[:current_points] - candidate)**2, axis=1))
            if np.min(distances) < min_distance:
                continue
        
        # Add the point
        points[current_points] = candidate
        current_points += 1
        
        # Log progress
        if current_points % 100 == 0:
            logger.info(f"Generated {current_points}/{num_points} nuclear coordinates")
    
    return points

def apply_real_transform(points_10x, magnification=4.0, rotation_degrees=5.0, 
                       translation=(100, 150), noise_level=2.0):
    """
    Apply a realistic transformation to simulate 10x to 40x mapping.
    
    Args:
        points_10x: Source points (10x)
        magnification: Scale factor between 10x and 40x (typically ~4)
        rotation_degrees: Rotation in degrees
        translation: Translation vector (tx, ty)
        noise_level: Gaussian noise standard deviation
        
    Returns:
        points_40x: Transformed points with noise (simulating 40x)
    """
    # Convert rotation to radians
    theta = np.radians(rotation_degrees)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    
    # Create transformation matrix
    scale_x = scale_y = magnification
    tx, ty = translation
    
    # [a, b, tx, c, d, ty] format
    params = np.array([
        scale_x * cos_theta, 
        -scale_y * sin_theta, 
        tx,
        scale_x * sin_theta, 
        scale_y * cos_theta, 
        ty
    ])
    
    # Apply transformation
    points_40x = np.zeros_like(points_10x)
    points_40x[:, 0] = (params[0] * points_10x[:, 0] + 
                        params[1] * points_10x[:, 1] + 
                        params[2])
    points_40x[:, 1] = (params[3] * points_10x[:, 0] + 
                        params[4] * points_10x[:, 1] + 
                        params[5])
    
    # Add noise to simulate real data
    if noise_level > 0:
        points_40x += np.random.normal(0, noise_level, points_40x.shape)
    
    return points_40x, params

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

def run_single_benchmark(trial_num, num_points, noise_level, num_iterations):
    """
    Run a single benchmark trial comparing L-BFGS-B and LMA.
    
    Args:
        trial_num: Trial number for tracking
        num_points: Number of points to generate
        noise_level: Noise level to apply
        num_iterations: Maximum optimizer iterations
        
    Returns:
        results: List of result dictionaries for the trial
    """
    logger.info(f"Starting trial {trial_num} with {num_points} points, noise={noise_level}")
    
    # Generate synthetic nuclear centroids (10x)
    points_10x = generate_nuclear_coordinates(num_points)
    
    # Apply transformation with noise to get simulated 40x points
    true_params = np.array([
        3.98, -0.3, 120.0,
        0.3, 3.98, 160.0
    ])
    points_40x, _ = apply_real_transform(
        points_10x, 
        magnification=4.0,
        rotation_degrees=5.0,
        translation=(120, 160),
        noise_level=noise_level
    )
    
    # Use same initial parameters for both optimizers
    initial_params = get_initial_params()
    
    # Run L-BFGS-B
    lbfgsb_params, lbfgsb_time, lbfgsb_memory, lbfgsb_iter, lbfgsb_rmsd = optimize_lbfgsb(
        points_10x, points_40x, initial_params.copy(), max_iter=num_iterations
    )
    
    # Run LMA
    lma_params, lma_time, lma_memory, lma_iter, lma_rmsd = optimize_lma(
        points_10x, points_40x, initial_params.copy(), max_iter=num_iterations
    )
    
    # Calculate error relative to true parameters
    lbfgsb_param_error = np.linalg.norm(lbfgsb_params - true_params)
    lma_param_error = np.linalg.norm(lma_params - true_params)
    
    # Store results
    results = []
    results.append({
        'trial': trial_num,
        'num_points': num_points,
        'noise_level': noise_level,
        'optimization': 'L-BFGS-B',
        'rmsd': lbfgsb_rmsd,
        'param_error': lbfgsb_param_error,
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
    
    results.append({
        'trial': trial_num,
        'num_points': num_points,
        'noise_level': noise_level,
        'optimization': 'LMA',
        'rmsd': lma_rmsd,
        'param_error': lma_param_error,
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

def run_parallel_benchmarks(num_trials, num_points, noise_levels, num_iterations, output_dir):
    """
    Run benchmarks in parallel across trials and noise levels.
    
    Args:
        num_trials: Number of trials to run for each configuration
        num_points: List of point counts to test
        noise_levels: List of noise levels to test
        num_iterations: Maximum optimizer iterations
        output_dir: Directory to save results
        
    Returns:
        results_df: DataFrame with benchmark results
    """
    logger.info("Starting parallel benchmark...")
    results_dir = Path(output_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create list of all benchmark configurations
    benchmark_configs = []
    trial_id = 0
    
    for n_points in num_points:
        for noise in noise_levels:
            for trial in range(num_trials):
                benchmark_configs.append((trial_id, n_points, noise, num_iterations))
                trial_id += 1
    
    # Determine optimal number of workers (leave one core free)
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    logger.info(f"Running {len(benchmark_configs)} benchmark configurations on {num_cores} CPU cores")
    
    # Run benchmarks in parallel
    results = []
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit all benchmark configurations
        futures = [executor.submit(run_single_benchmark, *config) for config in benchmark_configs]
        
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
    csv_path = results_dir / 'optimizer_benchmark_synthetic.csv'
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")
    
    # Create summary by optimizer and noise level
    summary = results_df.groupby(['optimization', 'noise_level', 'num_points']).agg({
        'rmsd': ['mean', 'std'],
        'param_error': ['mean', 'std'],
        'time_ms': ['mean', 'std'],
        'memory_kb': ['mean', 'std'],
        'iterations': ['mean', 'std']
    })
    
    summary_path = results_dir / 'optimizer_benchmark_synthetic_summary.csv'
    summary.to_csv(summary_path)
    logger.info(f"Summary saved to {summary_path}")
    
    # Plot results
    plot_dir = results_dir / 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot RMSD vs noise level for both optimizers
    for n_points in num_points:
        subset = results_df[results_df['num_points'] == n_points]
        plt.figure(figsize=(10, 6))
        for opt in ['L-BFGS-B', 'LMA']:
            opt_data = subset[subset['optimization'] == opt]
            mean_rmsd = opt_data.groupby('noise_level')['rmsd'].mean()
            std_rmsd = opt_data.groupby('noise_level')['rmsd'].std()
            
            plt.errorbar(
                mean_rmsd.index, 
                mean_rmsd.values, 
                yerr=std_rmsd.values, 
                label=opt, 
                marker='o',
                capsize=4
            )
        
        plt.xlabel('Noise Level')
        plt.ylabel('RMSD')
        plt.title(f'RMSD vs Noise Level ({n_points} points)')
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_dir / f'rmsd_vs_noise_{n_points}points.png')
    
    # Plot time comparison
    for n_points in num_points:
        subset = results_df[results_df['num_points'] == n_points]
        plt.figure(figsize=(10, 6))
        for opt in ['L-BFGS-B', 'LMA']:
            opt_data = subset[subset['optimization'] == opt]
            mean_time = opt_data.groupby('noise_level')['time_ms'].mean()
            std_time = opt_data.groupby('noise_level')['time_ms'].std()
            
            plt.errorbar(
                mean_time.index, 
                mean_time.values, 
                yerr=std_time.values, 
                label=opt, 
                marker='o',
                capsize=4
            )
        
        plt.xlabel('Noise Level')
        plt.ylabel('Time (ms)')
        plt.title(f'Computation Time vs Noise Level ({n_points} points)')
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_dir / f'time_vs_noise_{n_points}points.png')
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Parallel benchmark for L-BFGS-B vs LMA algorithms on synthetic data')
    parser.add_argument('--output-dir', type=str, default='results/optimizer_benchmark_synthetic', 
                        help='Directory to save results')
    parser.add_argument('--trials', type=int, default=5, 
                        help='Number of trials to run for each configuration')
    parser.add_argument('--points', type=str, default='100,500,1000', 
                        help='Comma-separated list of point counts to test')
    parser.add_argument('--noise', type=str, default='0,1,2,5,10', 
                        help='Comma-separated list of noise levels to test')
    parser.add_argument('--iterations', type=int, default=100, 
                        help='Maximum optimizer iterations')
    args = parser.parse_args()
    
    # Track total execution time
    total_start_time = time.time()
    
    # Display CPU information
    num_cores = multiprocessing.cpu_count()
    logger.info(f"Running on system with {num_cores} CPU cores")
    
    # Parse point counts and noise levels
    num_points = [int(x) for x in args.points.split(',')]
    noise_levels = [float(x) for x in args.noise.split(',')]
    
    # Run benchmarks in parallel
    run_parallel_benchmarks(
        args.trials, 
        num_points,
        noise_levels,
        args.iterations,
        args.output_dir
    )
    
    # Calculate total execution time
    total_time = time.time() - total_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Synthetic data benchmark completed successfully!")
    logger.info(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

if __name__ == "__main__":
    main()