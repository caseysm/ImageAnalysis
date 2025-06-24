#!/usr/bin/env python3
"""
Simplified benchmark script to compare L-BFGS-B and Levenberg-Marquardt algorithms
using synthetic point sets that mimic 10x and 40x nuclear coordinates.

This script:
1. Generates synthetic point sets mimicking nuclear coordinates
2. Applies known transformations to simulate 10x to 40x mapping
3. Evaluates both L-BFGS-B and LMA optimization algorithms
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('benchmark')

def generate_nuclear_coordinates(num_points=100, width=512, height=512, min_distance=20):
    """
    Generate synthetic nuclear coordinates with minimum distance constraints
    to mimic real nuclear distributions.

    Args:
        num_points: Number of nuclei to generate
        width: Image width
        height: Image height
        min_distance: Minimum distance between nuclei

    Returns:
        coordinates: Array of (y, x) coordinates
    """
    coordinates = []
    attempts = 0
    max_attempts = num_points * 100

    while len(coordinates) < num_points and attempts < max_attempts:
        # Generate random point
        x = np.random.randint(20, width - 20)
        y = np.random.randint(20, height - 20)

        # Check distance to existing points
        valid = True
        for cx, cy in coordinates:
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            if dist < min_distance:
                valid = False
                break

        if valid:
            coordinates.append((x, y))

        attempts += 1

    logger.info(f"Generated {len(coordinates)} nuclear coordinates in {attempts} attempts")
    # Return as float64 array to avoid type issues
    return np.array(coordinates, dtype=np.float64)

def apply_real_transform(points_10x, magnification=4.0, rotation_degrees=5.0, 
                        translation=(100, 150), noise_level=2.0):
    """
    Apply a realistic transformation to simulate 10x to 40x mapping.
    
    Args:
        points_10x: Array of 10x coordinates
        magnification: Scale factor (typically ~4 for 10x to 40x)
        rotation_degrees: Rotation angle in degrees
        translation: Translation vector (tx, ty)
        noise_level: Standard deviation of Gaussian noise
        
    Returns:
        points_40x: Transformed coordinates
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
    points_40x = transform_points(points_10x, params)

    # Convert to float64 before adding noise
    points_40x = points_40x.astype(np.float64)

    # Add Gaussian noise
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

def run_benchmark(output_dir, num_trials=3, point_counts=[50, 100, 200, 500, 1000]):
    """
    Run benchmark comparing L-BFGS-B and LMA algorithms.
    
    Args:
        output_dir: Directory to save results
        num_trials: Number of trials to run
        point_counts: List of point counts to test
        
    Returns:
        results_df: DataFrame with benchmark results
    """
    logger.info("Starting benchmark...")
    results_dir = Path(output_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    results = []
    true_params_list = []
    
    for num_points in point_counts:
        logger.info(f"Testing with {num_points} points")
        
        for trial in range(num_trials):
            logger.info(f"Trial {trial+1}/{num_trials}")
            
            # Generate random coordinates mimicking nuclear positions
            points_10x = generate_nuclear_coordinates(num_points=num_points)
            
            # Apply realistic transformation to create 40x points
            points_40x, true_params = apply_real_transform(
                points_10x,
                magnification=4.0,
                rotation_degrees=np.random.uniform(-10, 10),
                translation=(np.random.uniform(50, 150), np.random.uniform(50, 150)),
                noise_level=2.0
            )
            
            true_params_list.append({
                'num_points': num_points,
                'trial': trial,
                'a': true_params[0],
                'b': true_params[1],
                'tx': true_params[2],
                'c': true_params[3],
                'd': true_params[4],
                'ty': true_params[5]
            })
            
            # Get initial parameters for optimization
            initial_params = get_initial_params()
            
            # Run L-BFGS-B
            lbfgsb_params, lbfgsb_time, lbfgsb_memory, lbfgsb_iter, lbfgsb_rmsd = optimize_lbfgsb(
                points_10x, points_40x, initial_params.copy()
            )
            
            # Run LMA
            lma_params, lma_time, lma_memory, lma_iter, lma_rmsd = optimize_lma(
                points_10x, points_40x, initial_params.copy()
            )
            
            # Calculate parameter errors
            lbfgsb_param_error = np.mean(np.abs(lbfgsb_params - true_params))
            lma_param_error = np.mean(np.abs(lma_params - true_params))
            
            # Store results
            results.append({
                'num_points': num_points,
                'trial': trial,
                'optimization': 'L-BFGS-B',
                'rmsd': lbfgsb_rmsd,
                'time_ms': lbfgsb_time * 1000,  # Convert to milliseconds
                'memory_kb': lbfgsb_memory,
                'iterations': lbfgsb_iter,
                'param_error': lbfgsb_param_error,
                'a': lbfgsb_params[0],
                'b': lbfgsb_params[1],
                'tx': lbfgsb_params[2],
                'c': lbfgsb_params[3],
                'd': lbfgsb_params[4],
                'ty': lbfgsb_params[5]
            })
            
            results.append({
                'num_points': num_points,
                'trial': trial,
                'optimization': 'LMA',
                'rmsd': lma_rmsd,
                'time_ms': lma_time * 1000,  # Convert to milliseconds
                'memory_kb': lma_memory,
                'iterations': lma_iter,
                'param_error': lma_param_error,
                'a': lma_params[0],
                'b': lma_params[1],
                'tx': lma_params[2],
                'c': lma_params[3],
                'd': lma_params[4],
                'ty': lma_params[5]
            })
    
    # Create DataFrames
    results_df = pd.DataFrame(results)
    params_df = pd.DataFrame(true_params_list)
    
    # Save results to CSV
    results_df.to_csv(results_dir / 'optimizer_benchmark_synthetic.csv', index=False)
    params_df.to_csv(results_dir / 'true_params_synthetic.csv', index=False)
    logger.info(f"Results saved to {results_dir}")
    
    # Calculate summary statistics
    summary = results_df.groupby(['num_points', 'optimization']).agg({
        'rmsd': ['mean', 'std'],
        'time_ms': ['mean', 'std'],
        'memory_kb': ['mean', 'std'],
        'iterations': ['mean', 'std'],
        'param_error': ['mean', 'std']
    })
    
    summary.to_csv(results_dir / 'optimizer_benchmark_synthetic_summary.csv')
    
    # Create plots
    plot_dir = results_dir / 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot RMSD vs Point Count
    plt.figure(figsize=(10, 6))
    for opt in ['L-BFGS-B', 'LMA']:
        subset = results_df[results_df['optimization'] == opt]
        point_means = subset.groupby('num_points')['rmsd'].mean()
        point_stds = subset.groupby('num_points')['rmsd'].std()
        
        x = list(point_means.index)
        y = list(point_means.values)
        err = list(point_stds.values)
        
        plt.errorbar(x, y, yerr=err, fmt='o-' if opt == 'L-BFGS-B' else 's--', 
                    label=opt, capsize=5)
    
    plt.xlabel('Number of Points')
    plt.ylabel('RMSD')
    plt.title('RMSD vs. Number of Points')
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_dir / 'rmsd_vs_points.png')
    
    # Plot Computation Time vs Point Count
    plt.figure(figsize=(10, 6))
    for opt in ['L-BFGS-B', 'LMA']:
        subset = results_df[results_df['optimization'] == opt]
        point_means = subset.groupby('num_points')['time_ms'].mean()
        point_stds = subset.groupby('num_points')['time_ms'].std()
        
        x = list(point_means.index)
        y = list(point_means.values)
        err = list(point_stds.values)
        
        plt.errorbar(x, y, yerr=err, fmt='o-' if opt == 'L-BFGS-B' else 's--', 
                    label=opt, capsize=5)
    
    plt.xlabel('Number of Points')
    plt.ylabel('Computation Time (ms)')
    plt.title('Computation Time vs. Number of Points')
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_dir / 'time_vs_points.png')
    
    # Plot Parameter Error
    plt.figure(figsize=(10, 6))
    for opt in ['L-BFGS-B', 'LMA']:
        subset = results_df[results_df['optimization'] == opt]
        point_means = subset.groupby('num_points')['param_error'].mean()
        point_stds = subset.groupby('num_points')['param_error'].std()
        
        x = list(point_means.index)
        y = list(point_means.values)
        err = list(point_stds.values)
        
        plt.errorbar(x, y, yerr=err, fmt='o-' if opt == 'L-BFGS-B' else 's--', 
                    label=opt, capsize=5)
    
    plt.xlabel('Number of Points')
    plt.ylabel('Parameter Error')
    plt.title('Parameter Error vs. Number of Points')
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_dir / 'param_error_vs_points.png')
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Benchmark L-BFGS-B vs LMA algorithms with synthetic data')
    parser.add_argument('--output-dir', type=str, default='results/optimizer_benchmark_synthetic', 
                        help='Directory to save results')
    parser.add_argument('--trials', type=int, default=3, help='Number of trials to run')
    args = parser.parse_args()
    
    # Run benchmark
    run_benchmark(args.output_dir, num_trials=args.trials)
    
    logger.info("Benchmark completed successfully!")
    
if __name__ == "__main__":
    main()