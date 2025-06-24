#!/usr/bin/env python3
"""
This script adds additional visualizations to the large-scale benchmark results.
Run this after running the benchmark_optimizers_large_scale.py script.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('extra_visualizations')

def create_extra_visualizations(results_path, results_dir):
    """
    Generate additional visualization plots from benchmark results.
    
    Args:
        results_path: Path to the benchmark results CSV file
        results_dir: Directory to save plots
    """
    # Load results
    logger.info(f"Loading results from {results_path}")
    results_df = pd.read_csv(results_path)
    
    # Create plots directory
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_context("talk")
    sns.set_palette("colorblind")
    
    # Convert point counts to string for better categorical plotting
    results_df['points_label'] = results_df['num_points'].astype(str)
    
    logger.info("Creating additional visualization plots")
    
    # 8. Convergence Rate Analysis (iterations vs. dataset size)
    plt.figure(figsize=(12, 8))
    iterations_data = results_df.groupby(['num_points', 'optimization'])['iterations'].mean().reset_index()
    
    sns.lineplot(data=iterations_data, x='num_points', y='iterations', hue='optimization', 
               style='optimization', markers=True, dashes=False, linewidth=3, markersize=10)
    plt.xlabel('Number of Points')
    plt.ylabel('Average Iterations to Converge')
    plt.title('Convergence Rate vs. Dataset Size')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'convergence_rate.png'), dpi=300, bbox_inches='tight')
    
    # 9. Time-Accuracy Tradeoff (scatter plot with efficiency frontiers)
    plt.figure(figsize=(14, 10))
    
    # Plot all individual trials
    for opt in results_df['optimization'].unique():
        opt_data = results_df[results_df['optimization'] == opt]
        plt.scatter(opt_data['time_ms'], opt_data['rmsd'], 
                   alpha=0.6, label=opt, s=50)
        
    # Add diagonal lines representing "efficiency frontiers" (lower is better)
    time_min, time_max = results_df['time_ms'].min(), results_df['time_ms'].max()
    for efficiency in [0.01, 0.1, 1.0, 10.0, 100.0]:
        times = np.logspace(np.log10(time_min), np.log10(time_max), 100)
        plt.plot(times, efficiency * times, 'k--', alpha=0.3, linewidth=1)
        # Add label at the midpoint of each line
        midpoint_idx = len(times) // 2
        plt.text(times[midpoint_idx], efficiency * times[midpoint_idx], 
               f"Efficiency = {efficiency}", ha='center', va='bottom', 
               rotation=45, color='gray', alpha=0.7)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Computation Time (ms)')
    plt.ylabel('RMSD (pixels)')
    plt.title('Time-Accuracy Tradeoff')
    plt.legend(title='Optimizer')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'time_accuracy_tradeoff.png'), dpi=300, bbox_inches='tight')
    
    # 10. Parameter-Specific Errors
    # Group data by noise level, point count, and optimizer
    param_spec_data = results_df.groupby(['noise_level', 'num_points', 'optimization']).agg({
        'rmsd': 'mean',
        'param_error': 'mean',
        'time_ms': 'mean'
    }).reset_index()
    
    plt.figure(figsize=(14, 8))
    g = sns.FacetGrid(param_spec_data, col="noise_level", height=7, aspect=1.2)
    g.map_dataframe(sns.lineplot, x="num_points", y="param_error", hue="optimization", 
                  style="optimization", markers=True, dashes=False)
    g.set_axis_labels("Number of Points", "Parameter Error (pixels)")
    g.add_legend(title="Optimizer")
    g.set_titles("Noise Level: {col_name}")
    g.fig.suptitle("Parameter Error vs. Number of Points", y=1.05, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'parameter_specific_errors.png'), dpi=300, bbox_inches='tight')
    
    # 11. Heat Maps of RMSD vs noise level and outlier fraction
    pivot_data_lbfgsb = results_df[results_df['optimization'] == 'L-BFGS-B'].pivot_table(
        values='rmsd', index='noise_level', columns='outlier_fraction', aggfunc='mean')
    
    pivot_data_lma = results_df[results_df['optimization'] == 'LMA'].pivot_table(
        values='rmsd', index='noise_level', columns='outlier_fraction', aggfunc='mean')
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # L-BFGS-B heatmap
    sns.heatmap(pivot_data_lbfgsb, annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[0])
    axes[0].set_title('RMSD Heatmap: L-BFGS-B')
    axes[0].set_xlabel('Outlier Fraction')
    axes[0].set_ylabel('Noise Level')
    
    # LMA heatmap
    sns.heatmap(pivot_data_lma, annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[1])
    axes[1].set_title('RMSD Heatmap: LMA')
    axes[1].set_xlabel('Outlier Fraction')
    axes[1].set_ylabel('Noise Level')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'rmsd_heatmaps.png'), dpi=300, bbox_inches='tight')
    
    # 12. Performance Ratio Plot (LMA/L-BFGS-B)
    # Prepare data for ratio computation
    ratio_data = results_df.pivot_table(
        index=['num_points', 'noise_level', 'outlier_fraction', 'trial'],
        columns='optimization',
        values=['time_ms', 'rmsd', 'param_error', 'memory_kb']
    ).reset_index()
    
    # Compute ratios (LMA / L-BFGS-B)
    ratio_data['time_ratio'] = ratio_data[('time_ms', 'LMA')] / ratio_data[('time_ms', 'L-BFGS-B')]
    ratio_data['rmsd_ratio'] = ratio_data[('rmsd', 'LMA')] / ratio_data[('rmsd', 'L-BFGS-B')]
    ratio_data['memory_ratio'] = ratio_data[('memory_kb', 'LMA')] / ratio_data[('memory_kb', 'L-BFGS-B')]
    ratio_data['param_error_ratio'] = ratio_data[('param_error', 'LMA')] / ratio_data[('param_error', 'L-BFGS-B')]
    
    # Create aggregate by point count
    ratio_by_points = ratio_data.groupby('num_points').agg({
        'time_ratio': 'mean',
        'rmsd_ratio': 'mean',
        'memory_ratio': 'mean',
        'param_error_ratio': 'mean'
    }).reset_index()
    
    # Plot performance ratios
    plt.figure(figsize=(14, 8))
    
    metrics = ['time_ratio', 'rmsd_ratio', 'memory_ratio', 'param_error_ratio']
    labels = ['Computation Time', 'RMSD', 'Memory Usage', 'Parameter Error']
    
    for metric, label in zip(metrics, labels):
        plt.plot(ratio_by_points['num_points'], ratio_by_points[metric], 
                marker='o', linewidth=2, label=label)
        
    # Add a horizontal line at ratio=1 (equal performance)
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, 
                label='Equal Performance')
    
    plt.xlabel('Number of Points')
    plt.ylabel('Ratio (LMA / L-BFGS-B)')
    plt.title('Performance Ratio: LMA vs. L-BFGS-B')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'performance_ratio.png'), dpi=300, bbox_inches='tight')
    
    # 13. Success Rate Analysis
    # Define success thresholds for RMSD
    rmsd_thresholds = [
        results_df['rmsd'].quantile(0.25),  # Lower quartile as "excellent" threshold
        results_df['rmsd'].median(),       # Median as "good" threshold
        results_df['rmsd'].quantile(0.75)   # Upper quartile as "acceptable" threshold
    ]
    
    success_rates = []
    
    # Calculate success rates for each optimizer at each point count
    for optimizer in results_df['optimization'].unique():
        for point_count in results_df['num_points'].unique():
            subset = results_df[(results_df['optimization'] == optimizer) & 
                              (results_df['num_points'] == point_count)]
            
            total_trials = len(subset)
            for threshold_idx, threshold in enumerate(rmsd_thresholds):
                success_count = sum(subset['rmsd'] <= threshold)
                success_rate = success_count / total_trials if total_trials > 0 else 0
                
                success_rates.append({
                    'optimization': optimizer,
                    'num_points': point_count,
                    'threshold_idx': threshold_idx,
                    'threshold_value': threshold,
                    'success_rate': success_rate
                })
    
    success_df = pd.DataFrame(success_rates)
    
    # Create labels for the thresholds
    threshold_labels = ['Excellent', 'Good', 'Acceptable']
    success_df['threshold_label'] = success_df['threshold_idx'].apply(lambda x: threshold_labels[x])
    
    # Plot success rates
    plt.figure(figsize=(15, 9))
    g = sns.FacetGrid(success_df, col="threshold_label", height=7, aspect=1.1)
    g.map_dataframe(sns.lineplot, x="num_points", y="success_rate", hue="optimization", 
                   style="optimization", markers=True, dashes=False, linewidth=3, markersize=10)
    g.set_axis_labels("Number of Points", "Success Rate")
    g.add_legend(title="Optimizer")
    g.set_titles("Performance Threshold: {col_name}")
    g.fig.suptitle("Success Rate vs. Dataset Size", y=1.05, fontsize=16)
    
    # Add threshold values to the titles
    for i, ax in enumerate(g.axes.flatten()):
        threshold_value = rmsd_thresholds[i]
        title = ax.get_title()
        ax.set_title(f"{title}\n(RMSD ≤ {threshold_value:.3f} pixels)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'success_rate_analysis.png'), dpi=300, bbox_inches='tight')
    
    # 14. Robustness to Initial Conditions (across trials)
    # We'll use trial number as a proxy for different initializations
    robustness_data = results_df.groupby(['optimization', 'trial']).agg({
        'rmsd': ['mean', 'std'],
        'time_ms': ['mean', 'std']
    }).reset_index()
    
    # Calculate coefficient of variation (CV) for each optimizer across trials
    cv_data = []
    for optimizer in results_df['optimization'].unique():
        optimizer_data = robustness_data[robustness_data['optimization'] == optimizer]
        
        rmsd_mean = optimizer_data[('rmsd', 'mean')].mean()
        rmsd_std = optimizer_data[('rmsd', 'mean')].std()
        rmsd_cv = rmsd_std / rmsd_mean if rmsd_mean > 0 else 0
        
        time_mean = optimizer_data[('time_ms', 'mean')].mean() 
        time_std = optimizer_data[('time_ms', 'mean')].std()
        time_cv = time_std / time_mean if time_mean > 0 else 0
        
        cv_data.append({
            'optimization': optimizer,
            'rmsd_cv': rmsd_cv,
            'time_cv': time_cv
        })
    
    cv_df = pd.DataFrame(cv_data)
    
    # Plot coefficient of variation
    plt.figure(figsize=(12, 8))
    
    metrics = ['rmsd_cv', 'time_cv']
    labels = ['RMSD Variability', 'Computation Time Variability']
    
    x = range(len(cv_df))
    width = 0.35
    
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        plt.bar([pos + i*width for pos in x], cv_df[metric], width, 
               label=label, alpha=0.7)
    
    plt.xlabel('Optimizer')
    plt.ylabel('Coefficient of Variation\n(lower is more consistent)')
    plt.title('Optimizer Robustness to Initialization Conditions')
    plt.xticks([pos + width/2 for pos in x], cv_df['optimization'])
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'robustness_analysis.png'), dpi=300, bbox_inches='tight')
    
    # 15. Floating-Point Operation Count (FLOP) Estimation
    # We can approximate this based on iterations and problem size
    results_df['estimated_flops'] = results_df['iterations'] * results_df['num_points'] * 50  # Rough estimate
    
    plt.figure(figsize=(12, 8))
    flop_data = results_df.groupby(['num_points', 'optimization'])['estimated_flops'].mean().reset_index()
    
    sns.lineplot(data=flop_data, x='num_points', y='estimated_flops', hue='optimization', 
               style='optimization', markers=True, dashes=False, linewidth=3, markersize=10)
    plt.xlabel('Number of Points')
    plt.ylabel('Estimated FLOPs (logarithmic)')
    plt.title('Computational Complexity (Estimated FLOPs)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'computational_complexity.png'), dpi=300, bbox_inches='tight')
    
    logger.info(f"Additional visualization plots saved to {plots_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate additional visualizations for optimizer benchmark results')
    parser.add_argument('--results-dir', type=str, default='results/optimizer_benchmark_large_scale',
                       help='Directory containing benchmark results')
    parser.add_argument('--results-file', type=str, default='large_scale_benchmark.csv',
                       help='CSV file with benchmark results')
    args = parser.parse_args()
    
    results_path = os.path.join(args.results_dir, args.results_file)
    
    if not os.path.exists(results_path):
        logger.error(f"Results file not found: {results_path}")
        logger.error("Run benchmark_optimizers_large_scale.py first to generate results")
        return 1
    
    create_extra_visualizations(results_path, args.results_dir)
    logger.info("Additional visualizations created successfully")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())