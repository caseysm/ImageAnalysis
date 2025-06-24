# Optimizer Benchmark

This script benchmarks the L-BFGS-B and Levenberg-Marquardt (LMA) optimization algorithms for point set registration tasks, which is a critical component of our mapping pipeline.

## Overview

The script generates synthetic point sets and applies known transformations, then measures how effectively each algorithm can recover the original transformation parameters. It evaluates:

- **Accuracy**: Root Mean Square Deviation (RMSD) between transformed points and target points
- **Performance**: Computation time in milliseconds
- **Resource Usage**: Memory consumption in KB
- **Convergence**: Number of iterations required

## Usage

```bash
# Basic usage (5 trials per configuration)
python benchmark_optimizers.py

# Run with more trials and generate plots
python benchmark_optimizers.py --trials 10 --plot

# Specify custom output directory
python benchmark_optimizers.py --output-dir results/my_benchmark --plot
```

## Output

The script generates the following outputs:

1. **CSV Data**:
   - `optimizer_benchmark.csv`: Raw benchmark data for each trial
   - `optimizer_comparison_summary.csv`: Aggregated statistics and comparisons

2. **Plots** (if enabled with `--plot`):
   - `rmsd_vs_points.png`: Accuracy comparison at different point counts
   - `time_vs_points.png`: Computation time scaling with point count 
   - `memory_vs_points.png`: Memory usage scaling with point count
   - `error_vs_outliers.png`: Parameter error sensitivity to outliers

## Test Parameters

The benchmark evaluates both optimizers under various conditions:

- **Point Counts**: 50, 100, 200, 500, 1000 points
- **Noise Levels**: 0.01, 0.05, 0.1 (standard deviation relative to point range)
- **Outlier Fractions**: 0, 0.1, 0.2 (percentage of incorrect correspondences)
- **Transformations**: Various combinations of translation, rotation, and scaling

## Analysis Tips

When analyzing the results, consider:

1. **RMSD**: Lower is better, indicates more accurate transformation recovery
2. **Parameter Error**: Measures how close the recovered parameters are to the ground truth 
3. **Performance vs. Accuracy Tradeoff**: L-BFGS-B is often faster but may be less accurate in some cases
4. **Outlier Sensitivity**: How algorithm performance degrades as outlier percentage increases

## Implementation Notes

- Both algorithms start from the same initial transformation parameters
- Memory measurements are taken using Python's `tracemalloc` module
- All tests include synthetic noise to simulate real-world conditions