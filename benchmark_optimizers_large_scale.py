#!/usr/bin/env python3
"""
Large-scale benchmark comparing L-BFGS-B and Levenberg-Marquardt algorithms
for transformation optimization with varying dataset sizes (100-100,000 cells).

This script:
1. Generates synthetic point datasets of various sizes (100 to 100,000 points)
2. Applies known transformations with configurable noise levels
3. Benchmarks optimizer performance (L-BFGS-B vs LMA)
4. Measures and compares:
   - Time performance (ms)
   - Memory usage (KB)
   - Accuracy (RMSD and parameter error)
5. Creates visualization plots for analysis

Results are saved to CSV files and visualizations.
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
import psutil
import gc
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import seaborn as sns
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('large_scale_benchmark')

def generate_synthetic_data(num_points, noise_level=0.05, outlier_fraction=0.1,
                           translation=(10, 20), rotation=15, scale=(0.9, 1.1),
                           random_seed=None, simulate_magnification=False):
    """
    Generate synthetic point sets for testing transformation optimization.
    Optimized for large datasets.

    This function can generate data in two ways:
    1. Standard synthetic data with controlled transformation
    2. Simulated 10x/40x microscopy data (when simulate_magnification=True)

    The simulated microscopy data mimics how nuclear centroids would appear at different
    magnifications (10x and 40x), similar to the actual use case in the imageanalysis package.

    Args:
        num_points: Number of points to generate
        noise_level: Standard deviation of Gaussian noise
        outlier_fraction: Fraction of points that are outliers
        translation: Translation vector (tx, ty)
        rotation: Rotation angle in degrees
        scale: Scale factors (sx, sy)
        random_seed: Random seed for reproducibility
        simulate_magnification: If True, simulates 10x/40x microscopy data with ~4x scale difference

    Returns:
        src_points: Source points (Nx2) - represents 10x data when simulate_magnification=True
        dst_points: Destination points (Nx2) - represents 40x data when simulate_magnification=True
        true_params: True transformation parameters
    """
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    if simulate_magnification:
        # Simulate nuclear centroids as they would appear at 10x magnification
        # Generate random points in a typical microscopy field of view
        # For 10x, a typical field might be ~1200x1200 pixels
        src_points = np.random.rand(num_points, 2) * 1200  # 10x points (low-res)

        # Real microscopy parameters:
        # - 10x to 40x has approximately 4x scale difference
        # - There's typically some rotation between images (~1-5 degrees)
        # - Translation can be substantial due to stage movement
        magnification_scale = 4.0  # 40x is 4 times higher resolution than 10x
        microscopy_rotation = np.random.uniform(-5, 5)  # Small rotation between images
        microscopy_translation = (np.random.uniform(-100, 100), np.random.uniform(-100, 100))

        # Create true transformation parameters for 10x to 40x mapping
        theta = np.radians(microscopy_rotation)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        sx, sy = magnification_scale, magnification_scale  # 4x scale in both dimensions

        # True parameters: [sx*cos(θ), -sy*sin(θ), tx, sx*sin(θ), sy*cos(θ), ty]
        true_params = np.array([
            sx * cos_theta, -sy * sin_theta, microscopy_translation[0],
            sx * sin_theta, sy * cos_theta, microscopy_translation[1]
        ])

    else:
        # Standard synthetic data with specified parameters
        # Generate random source points efficiently using vectorized operations
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

    # Apply transformation to create destination points (vectorized)
    dst_points = np.zeros_like(src_points)
    dst_points[:, 0] = true_params[0] * src_points[:, 0] + true_params[1] * src_points[:, 1] + true_params[2]
    dst_points[:, 1] = true_params[3] * src_points[:, 0] + true_params[4] * src_points[:, 1] + true_params[5]

    # Add Gaussian noise to all points
    # The noise is proportional to the magnitude of the points - mimicking microscopy measurement error
    # For microscopy data, 40x (higher resolution) typically has less noise proportionally
    if simulate_magnification:
        # Less noise for 40x (higher resolution) data
        noise_scale = noise_level * 0.75 if simulate_magnification else noise_level
    else:
        noise_scale = noise_level

    dst_points += np.random.normal(0, noise_scale * np.max(dst_points), dst_points.shape)

    # Add outliers by replacing random points with random displacements
    # Outliers represent segmentation errors, detected cells that don't match, etc.
    num_outliers = int(num_points * outlier_fraction)
    if num_outliers > 0:
        outlier_indices = np.random.choice(num_points, num_outliers, replace=False)
        if simulate_magnification:
            # For microscopy data, outliers should be within reasonable image bounds
            dst_points[outlier_indices] = np.random.rand(num_outliers, 2) * 4800  # 40x field size
        else:
            dst_points[outlier_indices] = np.random.rand(num_outliers, 2) * 100

    return src_points, dst_points, true_params

def transform_points(points, params):
    """
    Apply transformation to points (vectorized for large datasets).
    
    Args:
        points: Array of points (Nx2)
        params: Transformation parameters [a, b, tx, c, d, ty]
        
    Returns:
        transformed_points: Transformed points (Nx2)
    """
    transformed = np.zeros_like(points)
    transformed[:, 0] = params[0] * points[:, 0] + params[1] * points[:, 1] + params[2]
    transformed[:, 1] = params[3] * points[:, 0] + params[4] * points[:, 1] + params[5]
    return transformed

def calculate_rmsd(src_points, dst_points, params):
    """
    Calculate Root Mean Square Deviation between transformed source points and destination points.
    Memory-efficient implementation for large datasets.
    
    Args:
        src_points: Source points (Nx2)
        dst_points: Destination points (Nx2)
        params: Transformation parameters
        
    Returns:
        rmsd: Root Mean Square Deviation
    """
    transformed = transform_points(src_points, params)
    # Calculate squared differences efficiently
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
    Get initial transformation parameters for optimization.
    
    Returns:
        params: Initial transformation parameters
    """
    # Start with identity transformation + small random perturbation
    return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]) + np.random.randn(6) * 0.1

def track_memory_usage():
    """
    Get current memory usage of the process.
    
    Returns:
        memory_usage: Memory usage in kilobytes
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024  # Convert to KB

def optimize_lbfgsb(src_points, dst_points, initial_params, max_iter=100):
    """
    Optimize transformation using L-BFGS-B algorithm.
    Memory usage tracking optimized for large datasets.
    
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
        rmsd: Final RMSD
    """
    memory_before = track_memory_usage()
    start_time = time.time()
    
    result = minimize(
        lbfgsb_objective,
        initial_params,
        args=(src_points, dst_points),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'disp': False}
    )
    
    end_time = time.time()
    memory_after = track_memory_usage()
    
    time_taken = end_time - start_time
    # Calculate memory usage delta
    memory_delta = memory_after - memory_before
    
    # Calculate final RMSD
    rmsd = calculate_rmsd(src_points, dst_points, result.x)
    
    return result.x, time_taken, memory_delta, result.nit, rmsd

def optimize_lma(src_points, dst_points, initial_params, max_iter=100):
    """
    Optimize transformation using Levenberg-Marquardt algorithm.
    Memory usage tracking optimized for large datasets.
    
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
        rmsd: Final RMSD
    """
    memory_before = track_memory_usage()
    start_time = time.time()
    
    result = least_squares(
        lma_objective,
        initial_params,
        args=(src_points, dst_points),
        method='lm',
        max_nfev=max_iter
    )
    
    end_time = time.time()
    memory_after = track_memory_usage()
    
    time_taken = end_time - start_time
    # Calculate memory usage delta
    memory_delta = memory_after - memory_before
    
    # Calculate final RMSD
    rmsd = calculate_rmsd(src_points, dst_points, result.x)
    
    return result.x, time_taken, memory_delta, result.nfev, rmsd

def run_benchmark_trial(point_count, noise_level, outlier_fraction, trial_num, max_iter=100, simulate_microscopy=True):
    """
    Run a single benchmark trial for a specific configuration.

    Args:
        point_count: Number of points to generate
        noise_level: Noise level for the synthetic data
        outlier_fraction: Fraction of outliers
        trial_num: Trial number
        max_iter: Maximum iterations for optimizers
        simulate_microscopy: If True, uses simulated 10x/40x microscopy data

    Returns:
        results: Dictionary with benchmark results
    """
    logger.info(f"Running trial {trial_num} with {point_count} points, noise={noise_level}, outliers={outlier_fraction}")

    # Generate random transformation parameters for this trial (for non-microscopy simulation)
    rotation = 15 + np.random.randn() * 5
    translation = (np.random.randn() * 20, np.random.randn() * 20)
    scale = (0.9 + np.random.randn() * 0.1, 1.1 + np.random.randn() * 0.1)

    # Generate synthetic data with consistent seed for fair comparison
    random_seed = trial_num * 1000 + int(point_count)
    src_points, dst_points, true_params = generate_synthetic_data(
        num_points=point_count,
        noise_level=noise_level,
        outlier_fraction=outlier_fraction,
        translation=translation,
        rotation=rotation,
        scale=scale,
        random_seed=random_seed,
        simulate_magnification=simulate_microscopy
    )
    
    # Use same initial parameters for both optimizers
    initial_params = get_initial_params()
    
    # Collect results
    results = []
    
    # Force garbage collection to minimize interference
    gc.collect()
    
    # Run L-BFGS-B
    lbfgsb_params, lbfgsb_time, lbfgsb_memory, lbfgsb_iter, lbfgsb_rmsd = optimize_lbfgsb(
        src_points, dst_points, initial_params.copy(), max_iter=max_iter
    )
    
    # Calculate parameter error for L-BFGS-B
    lbfgsb_param_error = np.mean(np.abs(lbfgsb_params - true_params))
    
    # Force garbage collection
    gc.collect()
    
    # Run LMA
    lma_params, lma_time, lma_memory, lma_iter, lma_rmsd = optimize_lma(
        src_points, dst_points, initial_params.copy(), max_iter=max_iter
    )
    
    # Calculate parameter error for LMA
    lma_param_error = np.mean(np.abs(lma_params - true_params))
    
    # Store results for L-BFGS-B
    results.append({
        'num_points': point_count,
        'noise_level': noise_level,
        'outlier_fraction': outlier_fraction,
        'trial': trial_num,
        'optimization': 'L-BFGS-B',
        'rmsd': lbfgsb_rmsd,
        'time_ms': lbfgsb_time * 1000,  # Convert to milliseconds
        'memory_kb': lbfgsb_memory,
        'iterations': lbfgsb_iter,
        'param_error': lbfgsb_param_error
    })
    
    # Store results for LMA
    results.append({
        'num_points': point_count,
        'noise_level': noise_level,
        'outlier_fraction': outlier_fraction,
        'trial': trial_num,
        'optimization': 'LMA',
        'rmsd': lma_rmsd,
        'time_ms': lma_time * 1000,  # Convert to milliseconds
        'memory_kb': lma_memory,
        'iterations': lma_iter,
        'param_error': lma_param_error
    })
    
    return results

def run_parallel_benchmarks(point_counts, noise_levels, outlier_fractions, num_trials, max_iter, results_dir, simulate_microscopy=True):
    """
    Run benchmarks in parallel for multiple configurations.

    Args:
        point_counts: List of point counts to benchmark
        noise_levels: List of noise levels to test
        outlier_fractions: List of outlier fractions to test
        num_trials: Number of trials per configuration
        max_iter: Maximum iterations for optimizers
        results_dir: Directory to save results
        simulate_microscopy: If True, simulates 10x/40x microscopy data

    Returns:
        results_df: DataFrame with benchmark results
    """
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Create benchmark configurations
    benchmark_configs = []
    for point_count in point_counts:
        for noise_level in noise_levels:
            for outlier_fraction in outlier_fractions:
                for trial in range(num_trials):
                    benchmark_configs.append((point_count, noise_level, outlier_fraction, trial, max_iter, simulate_microscopy))
    
    total_benchmarks = len(benchmark_configs)
    logger.info(f"Running {total_benchmarks} benchmark configurations")
    
    # Determine number of workers (leave one core free for system)
    import multiprocessing
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    logger.info(f"Using {num_cores} CPU cores for parallel processing")
    
    # Run benchmarks in parallel with progress tracking
    all_results = []
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit all tasks
        future_to_config = {
            executor.submit(run_benchmark_trial, *config): config 
            for config in benchmark_configs
        }
        
        # Process results as they complete with progress bar
        for i, future in enumerate(tqdm(
            future_to_config, 
            total=len(future_to_config),
            desc="Running benchmarks"
        )):
            config = future_to_config[future]
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error in benchmark {config}: {e}")
    
    # Create DataFrame from all results
    results_df = pd.DataFrame(all_results)
    
    # Save raw results
    csv_path = os.path.join(results_dir, 'large_scale_benchmark.csv')
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Raw results saved to {csv_path}")
    
    # Create summary statistics grouped by optimizer, point count, and noise level
    summary = results_df.groupby(['optimization', 'num_points', 'noise_level', 'outlier_fraction']).agg({
        'rmsd': ['mean', 'std'],
        'time_ms': ['mean', 'std'],
        'memory_kb': ['mean', 'std'],
        'iterations': ['mean', 'std'],
        'param_error': ['mean', 'std']
    }).reset_index()
    
    # Save summary
    summary_path = os.path.join(results_dir, 'large_scale_benchmark_summary.csv')
    summary.to_csv(summary_path)
    logger.info(f"Summary results saved to {summary_path}")
    
    return results_df

def create_visualizations(results_df, results_dir):
    """
    Generate visualization plots from benchmark results.
    
    Args:
        results_df: DataFrame with benchmark results
        results_dir: Directory to save plots
    """
    # Create plots directory
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_context("talk")
    sns.set_palette("colorblind")
    
    # Convert point counts to string for better categorical plotting
    results_df['points_label'] = results_df['num_points'].astype(str)
    
    # 1. Computation Time vs. Number of Points (by optimizer)
    plt.figure(figsize=(16, 10))
    time_data = results_df.groupby(['num_points', 'optimization', 'noise_level'])['time_ms'].mean().reset_index()
    
    g = sns.FacetGrid(time_data, col="noise_level", height=7, aspect=1.2, sharey=False)
    g.map_dataframe(sns.lineplot, x="num_points", y="time_ms", hue="optimization", style="optimization", 
                    markers=True, dashes=False, linewidth=3, markersize=10)
    g.set_axis_labels("Number of Points", "Computation Time (ms)")
    g.add_legend(title="Optimizer")
    g.set_titles("Noise Level: {col_name}")
    g.fig.suptitle("Computation Time vs. Number of Points", y=1.05, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'time_vs_points.png'), dpi=300, bbox_inches='tight')
    
    # 2. RMSD vs. Number of Points (by optimizer and noise level)
    plt.figure(figsize=(16, 10))
    rmsd_data = results_df.groupby(['num_points', 'optimization', 'noise_level'])['rmsd'].mean().reset_index()

    g = sns.FacetGrid(rmsd_data, col="noise_level", height=7, aspect=1.2, sharey=False)
    g.map_dataframe(sns.lineplot, x="num_points", y="rmsd", hue="optimization", style="optimization",
                    markers=True, dashes=False, linewidth=3, markersize=10)
    g.set_axis_labels("Number of Points", "RMSD (pixels)")
    g.add_legend(title="Optimizer")
    g.set_titles("Noise Level: {col_name}")
    g.fig.suptitle("Accuracy (RMSD) vs. Number of Points", y=1.05, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'rmsd_vs_points.png'), dpi=300, bbox_inches='tight')
    
    # 3. Memory Usage vs. Number of Points (by optimizer)
    plt.figure(figsize=(16, 10))
    mem_data = results_df.groupby(['num_points', 'optimization'])['memory_kb'].mean().reset_index()
    
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=mem_data, x='num_points', y='memory_kb', hue='optimization', 
                style='optimization', markers=True, dashes=False, linewidth=3, markersize=10)
    plt.xlabel('Number of Points')
    plt.ylabel('Memory Usage (KB)')
    plt.title('Memory Usage vs. Number of Points')
    plt.xscale('log')  # Log scale for better visualization with large ranges
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'memory_vs_points.png'), dpi=300, bbox_inches='tight')
    
    # 4. Parameter Error vs. Outlier Fraction (by optimizer and noise level)
    plt.figure(figsize=(16, 10))
    error_data = results_df.groupby(['outlier_fraction', 'optimization', 'noise_level'])['param_error'].mean().reset_index()

    g = sns.FacetGrid(error_data, col="noise_level", height=7, aspect=1.2, sharey=False)
    g.map_dataframe(sns.lineplot, x="outlier_fraction", y="param_error", hue="optimization", style="optimization",
                    markers=True, dashes=False, linewidth=3, markersize=10)
    g.set_axis_labels("Outlier Fraction", "Parameter Error (pixels)")
    g.add_legend(title="Optimizer")
    g.set_titles("Noise Level: {col_name}")
    g.fig.suptitle("Parameter Error vs. Outlier Fraction", y=1.05, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'param_error_vs_outliers.png'), dpi=300, bbox_inches='tight')
    
    # 5. Computation Time by Point Count (log scale, boxplot)
    plt.figure(figsize=(16, 10))
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=results_df, x='points_label', y='time_ms', hue='optimization')
    plt.xlabel('Number of Points')
    plt.ylabel('Computation Time (ms)')
    plt.title('Computation Time Distribution by Point Count')
    plt.yscale('log')  # Log scale for better visualization with large ranges
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'time_boxplot.png'), dpi=300, bbox_inches='tight')
    
    # 6. Performance Comparison (barplot summarizing key metrics)
    # Create comparative metric: time per point
    results_df['time_per_point'] = results_df['time_ms'] / results_df['num_points']
    
    plt.figure(figsize=(16, 12))
    metrics = ['time_per_point', 'rmsd', 'memory_kb', 'param_error']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        comparison = results_df.groupby(['optimization'])[metric].mean().reset_index()
        sns.barplot(data=comparison, x='optimization', y=metric, ax=ax)
        ax.set_title(f'Average {metric}')
        ax.set_xlabel('')
        if metric in ['memory_kb', 'time_per_point']:
            ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    
    # 7. Scaling behavior: how metrics scale with point count
    plt.figure(figsize=(16, 10))
    
    scaling_data = results_df.groupby(['num_points', 'optimization']).agg({
        'time_ms': 'mean',
        'memory_kb': 'mean',
        'rmsd': 'mean'
    }).reset_index()
    
    # Normalize to scaling factor relative to smallest point count
    for opt in scaling_data['optimization'].unique():
        opt_data = scaling_data[scaling_data['optimization'] == opt]
        base_time = opt_data.loc[opt_data['num_points'].idxmin(), 'time_ms']
        base_memory = opt_data.loc[opt_data['num_points'].idxmin(), 'memory_kb']
        
        scaling_data.loc[scaling_data['optimization'] == opt, 'time_scale'] = scaling_data.loc[scaling_data['optimization'] == opt, 'time_ms'] / base_time
        scaling_data.loc[scaling_data['optimization'] == opt, 'memory_scale'] = scaling_data.loc[scaling_data['optimization'] == opt, 'memory_kb'] / base_memory
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Time scaling
    sns.lineplot(data=scaling_data, x='num_points', y='time_scale', hue='optimization', 
                style='optimization', markers=True, dashes=False, ax=axes[0])
    axes[0].set_title('Time Scaling vs. Point Count')
    axes[0].set_xlabel('Number of Points')
    axes[0].set_ylabel('Relative Time (normalized)')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    
    # Memory scaling
    sns.lineplot(data=scaling_data, x='num_points', y='memory_scale', hue='optimization', 
                style='optimization', markers=True, dashes=False, ax=axes[1])
    axes[1].set_title('Memory Scaling vs. Point Count')
    axes[1].set_xlabel('Number of Points')
    axes[1].set_ylabel('Relative Memory (normalized)')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'scaling_behavior.png'), dpi=300, bbox_inches='tight')
    
    logger.info(f"Visualization plots saved to {plots_dir}")

def main():
    parser = argparse.ArgumentParser(description='Large-scale benchmark comparing L-BFGS-B and LMA optimizers')

    # Point counts to test (exponential range from 100 to 100,000)
    parser.add_argument('--point-counts', type=str, default='100,500,1000,5000,10000,50000,100000',
                        help='Comma-separated list of point counts to test')

    # Noise levels
    parser.add_argument('--noise-levels', type=str, default='0.01,0.05,0.1',
                        help='Comma-separated list of noise levels to test')

    # Outlier fractions
    parser.add_argument('--outlier-fractions', type=str, default='0,0.1,0.2',
                        help='Comma-separated list of outlier fractions to test')

    # Number of trials per configuration
    parser.add_argument('--trials', type=int, default=3,
                        help='Number of trials per configuration')

    # Maximum iterations for optimizers
    parser.add_argument('--max-iter', type=int, default=100,
                        help='Maximum optimizer iterations')

    # Results directory
    parser.add_argument('--results-dir', type=str, default='results/optimizer_benchmark_large_scale',
                        help='Directory to save benchmark results')

    # Option to skip the largest point counts for faster testing
    parser.add_argument('--quick-test', action='store_true',
                        help='Run only smaller point counts (≤10,000) for faster testing')

    # Option to use realistic microscopy simulation (10x/40x)
    parser.add_argument('--microscopy-simulation', action='store_true', default=True,
                        help='Simulate realistic 10x/40x microscopy data with 4x magnification difference')
    
    args = parser.parse_args()
    
    # Parse arguments
    point_counts = [int(x) for x in args.point_counts.split(',')]
    noise_levels = [float(x) for x in args.noise_levels.split(',')]
    outlier_fractions = [float(x) for x in args.outlier_fractions.split(',')]
    
    # If quick test, limit point counts
    if args.quick_test:
        point_counts = [p for p in point_counts if p <= 10000]
        logger.info("Running quick test with point counts: %s", point_counts)
    
    # Track total execution time
    start_time = time.time()
    
    # Log configuration
    logger.info("Starting large-scale optimizer benchmark with:")
    logger.info("  Point counts: %s", point_counts)
    logger.info("  Noise levels: %s", noise_levels)
    logger.info("  Outlier fractions: %s", outlier_fractions)
    logger.info("  Trials per configuration: %d", args.trials)
    logger.info("  Max iterations: %d", args.max_iter)
    logger.info("  Results directory: %s", args.results_dir)
    
    # Run benchmarks in parallel
    results_df = run_parallel_benchmarks(
        point_counts=point_counts,
        noise_levels=noise_levels,
        outlier_fractions=outlier_fractions,
        num_trials=args.trials,
        max_iter=args.max_iter,
        results_dir=args.results_dir,
        simulate_microscopy=args.microscopy_simulation
    )
    
    # Create visualizations
    create_visualizations(results_df, args.results_dir)
    
    # Calculate total execution time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Benchmark completed successfully!")
    logger.info(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    logger.info(f"Results saved to {args.results_dir}")

if __name__ == "__main__":
    main()