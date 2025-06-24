#!/usr/bin/env python3
"""
Benchmark script to compare L-BFGS-B and Levenberg-Marquardt algorithms 
for transformation optimization in point matching tasks.

Metrics compared:
- RMSD (Root Mean Square Deviation)
- Memory usage
- Computation time

Results are saved to a CSV file.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
import tracemalloc
from pathlib import Path
import argparse

def generate_test_data(num_points, noise_level=0.05, outlier_fraction=0.1, 
                       translation=(10, 20), rotation=15, scale=(0.9, 1.1)):
    """
    Generate synthetic point sets for testing transformation optimization.
    
    Args:
        num_points: Number of points to generate
        noise_level: Standard deviation of Gaussian noise
        outlier_fraction: Fraction of points that are outliers
        translation: Translation vector (tx, ty)
        rotation: Rotation angle in degrees
        scale: Scale factors (sx, sy)
        
    Returns:
        src_points: Source points (Nx2)
        dst_points: Destination points (Nx2)
        true_params: True transformation parameters
    """
    # Generate random source points
    src_points = np.random.rand(num_points, 2) * 100
    
    # Create true transformation parameters
    theta = np.radians(rotation)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    sx, sy = scale
    
    # True parameters: [sx*cos(θ), -sy*sin(θ), tx, sx*sin(θ), sy*cos(θ), ty]
    true_params = np.array([
        sx * cos_theta, -sy * sin_theta, translation[0],
        sx * sin_theta, sy * cos_theta, translation[1]
    ])
    
    # Apply transformation to create destination points
    dst_points = np.zeros_like(src_points)
    for i in range(num_points):
        x, y = src_points[i]
        dst_points[i, 0] = true_params[0] * x + true_params[1] * y + true_params[2]
        dst_points[i, 1] = true_params[3] * x + true_params[4] * y + true_params[5]
    
    # Add Gaussian noise to all points
    dst_points += np.random.normal(0, noise_level * np.max(dst_points), dst_points.shape)
    
    # Add outliers by replacing random points with random displacements
    num_outliers = int(num_points * outlier_fraction)
    if num_outliers > 0:
        outlier_indices = np.random.choice(num_points, num_outliers, replace=False)
        dst_points[outlier_indices] = np.random.rand(num_outliers, 2) * 100
    
    return src_points, dst_points, true_params

def transform_points(points, params):
    """
    Apply transformation to points.
    
    Args:
        points: Array of points (Nx2)
        params: Transformation parameters [a, b, tx, c, d, ty]
        
    Returns:
        transformed_points: Transformed points (Nx2)
    """
    transformed = np.zeros_like(points)
    for i in range(len(points)):
        x, y = points[i]
        transformed[i, 0] = params[0] * x + params[1] * y + params[2]
        transformed[i, 1] = params[3] * x + params[4] * y + params[5]
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

def get_initial_params():
    """
    Get initial transformation parameters for optimization.
    
    Returns:
        params: Initial transformation parameters
    """
    # Start with identity transformation + small random perturbation
    return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]) + np.random.randn(6) * 0.1

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

def optimize_lbfgsb(src_points, dst_points, initial_params, max_iter=100):
    """
    Optimize transformation using L-BFGS-B algorithm.
    
    Args:
        src_points: Source points
        dst_points: Destination points
        initial_params: Initial transformation parameters
        max_iter: Maximum number of iterations
        
    Returns:
        params: Optimized parameters
        time_taken: Computation time
        max_memory: Peak memory usage
        iterations: Number of iterations
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
    
    return result.x, time_taken, max_memory, result.nit

def optimize_lma(src_points, dst_points, initial_params, max_iter=100):
    """
    Optimize transformation using Levenberg-Marquardt algorithm.
    
    Args:
        src_points: Source points
        dst_points: Destination points
        initial_params: Initial transformation parameters
        max_iter: Maximum number of iterations
        
    Returns:
        params: Optimized parameters
        time_taken: Computation time
        max_memory: Peak memory usage
        iterations: Number of iterations
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
    
    return result.x, time_taken, max_memory, result.nfev

def run_benchmark(num_trials=10, point_counts=[50, 100, 200, 500, 1000], 
                 noise_levels=[0.01, 0.05, 0.1], outlier_fractions=[0, 0.1, 0.2],
                 results_dir='results'):
    """
    Run benchmark comparing L-BFGS-B and LMA algorithms.
    
    Args:
        num_trials: Number of trials to run for each configuration
        point_counts: List of point counts to test
        noise_levels: List of noise levels to test
        outlier_fractions: List of outlier fractions to test
        results_dir: Directory to save results
    
    Returns:
        results_df: DataFrame with benchmark results
    """
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Create results dataframe
    results = []
    
    # Parameters for generating test data
    translations = [(10, 20), (50, 50), (100, 150)]
    rotations = [5, 15, 30, 45]
    scales = [(0.9, 1.1), (0.8, 1.2), (1.2, 0.8)]
    
    total_tests = len(point_counts) * len(noise_levels) * len(outlier_fractions) * num_trials
    test_counter = 0
    
    print(f"Running {total_tests} benchmarks...")
    
    for num_points in point_counts:
        for noise_level in noise_levels:
            for outlier_fraction in outlier_fractions:
                for trial in range(num_trials):
                    test_counter += 1
                    print(f"Test {test_counter}/{total_tests}: {num_points} points, noise = {noise_level}, outliers = {outlier_fraction}, trial {trial+1}/{num_trials}")
                    
                    # Select random transformation parameters for this trial
                    translation = translations[trial % len(translations)]
                    rotation = rotations[trial % len(rotations)]
                    scale = scales[trial % len(scales)]
                    
                    # Generate test data
                    src_points, dst_points, true_params = generate_test_data(
                        num_points=num_points,
                        noise_level=noise_level,
                        outlier_fraction=outlier_fraction,
                        translation=translation,
                        rotation=rotation,
                        scale=scale
                    )
                    
                    # Use same initial parameters for both optimizers
                    initial_params = get_initial_params()
                    
                    # Run L-BFGS-B
                    lbfgsb_params, lbfgsb_time, lbfgsb_memory, lbfgsb_iter = optimize_lbfgsb(
                        src_points, dst_points, initial_params.copy()
                    )
                    
                    # Run LMA
                    lma_params, lma_time, lma_memory, lma_iter = optimize_lma(
                        src_points, dst_points, initial_params.copy()
                    )
                    
                    # Calculate RMSD for both methods
                    lbfgsb_rmsd = calculate_rmsd(src_points, dst_points, lbfgsb_params)
                    lma_rmsd = calculate_rmsd(src_points, dst_points, lma_params)
                    
                    # Calculate parameter error (mean absolute difference from true params)
                    lbfgsb_param_error = np.mean(np.abs(lbfgsb_params - true_params))
                    lma_param_error = np.mean(np.abs(lma_params - true_params))
                    
                    # Store results
                    results.append({
                        'num_points': num_points,
                        'noise_level': noise_level,
                        'outlier_fraction': outlier_fraction,
                        'trial': trial,
                        'optimization': 'L-BFGS-B',
                        'rmsd': lbfgsb_rmsd,
                        'time_ms': lbfgsb_time * 1000,  # Convert to milliseconds
                        'memory_kb': lbfgsb_memory,
                        'iterations': lbfgsb_iter,
                        'param_error': lbfgsb_param_error
                    })
                    
                    results.append({
                        'num_points': num_points,
                        'noise_level': noise_level,
                        'outlier_fraction': outlier_fraction,
                        'trial': trial,
                        'optimization': 'LMA',
                        'rmsd': lma_rmsd,
                        'time_ms': lma_time * 1000,  # Convert to milliseconds
                        'memory_kb': lma_memory,
                        'iterations': lma_iter,
                        'param_error': lma_param_error
                    })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    csv_path = os.path.join(results_dir, 'optimizer_benchmark.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    return results_df

def plot_results(results_df, results_dir='results'):
    """
    Generate plots comparing L-BFGS-B and LMA performance.
    
    Args:
        results_df: DataFrame with benchmark results
        results_dir: Directory to save plots
    """
    # Ensure plots directory exists
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Calculate average metrics for each configuration
    summary = results_df.groupby(['num_points', 'noise_level', 'outlier_fraction', 'optimization']).agg({
        'rmsd': 'mean',
        'time_ms': 'mean',
        'memory_kb': 'mean',
        'iterations': 'mean',
        'param_error': 'mean'
    }).reset_index()
    
    # Plot 1: RMSD vs Number of Points
    plt.figure(figsize=(12, 8))
    for noise in summary['noise_level'].unique():
        for outlier in summary['outlier_fraction'].unique():
            subset = summary[(summary['noise_level'] == noise) & 
                             (summary['outlier_fraction'] == outlier)]
            
            lbfgsb_data = subset[subset['optimization'] == 'L-BFGS-B']
            lma_data = subset[subset['optimization'] == 'LMA']
            
            label = f"Noise={noise}, Outliers={outlier}"
            plt.plot(lbfgsb_data['num_points'], lbfgsb_data['rmsd'], 
                     'o-', label=f"L-BFGS-B ({label})")
            plt.plot(lma_data['num_points'], lma_data['rmsd'], 
                     's--', label=f"LMA ({label})")
    
    plt.xlabel('Number of Points')
    plt.ylabel('RMSD')
    plt.title('RMSD vs Number of Points')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'rmsd_vs_points.png'))
    
    # Plot 2: Computation Time vs Number of Points
    plt.figure(figsize=(12, 8))
    for noise in [summary['noise_level'].min()]:  # Just use the minimum noise level for clarity
        for outlier in [0.0, summary['outlier_fraction'].max()]:  # Use 0 and max outlier
            subset = summary[(summary['noise_level'] == noise) & 
                             (summary['outlier_fraction'] == outlier)]
            
            lbfgsb_data = subset[subset['optimization'] == 'L-BFGS-B']
            lma_data = subset[subset['optimization'] == 'LMA']
            
            label = f"Noise={noise}, Outliers={outlier}"
            plt.plot(lbfgsb_data['num_points'], lbfgsb_data['time_ms'], 
                     'o-', label=f"L-BFGS-B ({label})")
            plt.plot(lma_data['num_points'], lma_data['time_ms'], 
                     's--', label=f"LMA ({label})")
    
    plt.xlabel('Number of Points')
    plt.ylabel('Computation Time (ms)')
    plt.title('Computation Time vs Number of Points')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'time_vs_points.png'))
    
    # Plot 3: Memory Usage vs Number of Points
    plt.figure(figsize=(12, 8))
    for noise in [summary['noise_level'].min()]:  # Just use the minimum noise level for clarity
        for outlier in [0.0]:  # Just use no outliers for clarity
            subset = summary[(summary['noise_level'] == noise) & 
                             (summary['outlier_fraction'] == outlier)]
            
            lbfgsb_data = subset[subset['optimization'] == 'L-BFGS-B']
            lma_data = subset[subset['optimization'] == 'LMA']
            
            plt.plot(lbfgsb_data['num_points'], lbfgsb_data['memory_kb'], 
                     'o-', label=f"L-BFGS-B")
            plt.plot(lma_data['num_points'], lma_data['memory_kb'], 
                     's--', label=f"LMA")
    
    plt.xlabel('Number of Points')
    plt.ylabel('Memory Usage (KB)')
    plt.title('Memory Usage vs Number of Points')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'memory_vs_points.png'))
    
    # Plot 4: Parameter Error vs Outlier Fraction
    plt.figure(figsize=(12, 8))
    for points in summary['num_points'].unique():
        if points in [100, 500, 1000]:  # Select a few point counts for clarity
            for noise in [0.01, 0.1]:  # Select a few noise levels for clarity
                subset = summary[(summary['num_points'] == points) & 
                                 (summary['noise_level'] == noise)]
                
                lbfgsb_data = subset[subset['optimization'] == 'L-BFGS-B']
                lma_data = subset[subset['optimization'] == 'LMA']
                
                label = f"Points={points}, Noise={noise}"
                plt.plot(lbfgsb_data['outlier_fraction'], lbfgsb_data['param_error'], 
                         'o-', label=f"L-BFGS-B ({label})")
                plt.plot(lma_data['outlier_fraction'], lma_data['param_error'], 
                         's--', label=f"LMA ({label})")
    
    plt.xlabel('Outlier Fraction')
    plt.ylabel('Parameter Error')
    plt.title('Parameter Error vs Outlier Fraction')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'error_vs_outliers.png'))
    
    # Summary stats
    comparison = summary.pivot_table(
        index=['num_points', 'noise_level', 'outlier_fraction'],
        columns='optimization',
        values=['rmsd', 'time_ms', 'memory_kb', 'iterations', 'param_error']
    )
    
    # Calculate ratios for easier comparison
    comparison['rmsd_ratio'] = comparison[('rmsd', 'LMA')] / comparison[('rmsd', 'L-BFGS-B')]
    comparison['time_ratio'] = comparison[('time_ms', 'LMA')] / comparison[('time_ms', 'L-BFGS-B')]
    comparison['memory_ratio'] = comparison[('memory_kb', 'LMA')] / comparison[('memory_kb', 'L-BFGS-B')]
    
    # Save comparison summary
    comparison.to_csv(os.path.join(results_dir, 'optimizer_comparison_summary.csv'))
    print(f"Comparison summary saved to {os.path.join(results_dir, 'optimizer_comparison_summary.csv')}")

def main():
    parser = argparse.ArgumentParser(description='Benchmark L-BFGS-B vs LMA algorithms')
    parser.add_argument('--trials', type=int, default=5, help='Number of trials per configuration')
    parser.add_argument('--output-dir', type=str, default='results/optimizer_benchmark', 
                        help='Directory to save results')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    args = parser.parse_args()
    
    # Run benchmarks
    results_df = run_benchmark(
        num_trials=args.trials, 
        point_counts=[50, 100, 200, 500, 1000],
        results_dir=args.output_dir
    )
    
    # Generate plots if requested
    if args.plot:
        plot_results(results_df, results_dir=args.output_dir)
    
    print("Benchmark completed successfully!")
    
if __name__ == "__main__":
    main()