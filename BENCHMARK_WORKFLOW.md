# Nuclear Centroids Mapping Benchmark Workflow

This document describes the workflow for running segmentation in parallel and benchmarking L-BFGS-B vs. LMA optimization algorithms on real nuclear coordinates.

## Overview

This workflow consists of two main steps:

1. **Parallel Segmentation**: Process all 10x and 40x microscopy images to extract nuclear centroids
2. **Optimizer Benchmark**: Compare the performance of L-BFGS-B and LMA algorithms on the extracted centroids

## Quickstart

To run the complete workflow in one step:

```bash
./run_full_benchmark.sh
```

This script will:
1. Run the segmentation pipeline in parallel using all available CPU cores
2. Extract nuclear centroids from the segmentation results
3. Run the optimizer benchmark on the extracted centroids
4. Generate comprehensive performance reports

## Step-by-Step Process

If you prefer to run the steps individually:

### Step 1: Segmentation

Run the segmentation pipeline in parallel on all available images:

```bash
./run_segmentation_parallel.sh
```

This script will:
- Find all 10x and 40x images in the data directory
- Process all images in parallel, utilizing all available CPU cores
- Extract nuclear centroids from segmentation results
- Save the results to `results/segmentation_output/`

### Step 2: Optimizer Benchmark

Run the optimizer benchmark on the extracted centroids:

```bash
./run_optimizer_benchmark.sh
```

This script will:
- Load nuclear centroids from the segmentation results
- Pair corresponding 10x and 40x images
- Run both L-BFGS-B and LMA optimization algorithms on the point sets
- Measure RMSD, time, and memory usage
- Generate comprehensive performance reports

## Result Examination

After running the full workflow, you can find the results in:

- **Segmentation Results**: `results/segmentation_output/`
  - 10x centroids: `results/segmentation_output/10X/`
  - 40x centroids: `results/segmentation_output/40X/`
  - Summary: `results/segmentation_output/segmentation_summary.json`

- **Benchmark Results**: `results/optimizer_benchmark/`
  - Raw data: `results/optimizer_benchmark/optimizer_benchmark_real_points.csv`
  - Summary by well: `results/optimizer_benchmark/optimizer_benchmark_summary.csv`
  - Overall summary: `results/optimizer_benchmark/optimizer_benchmark_overall.csv`
  - Visualizations: `results/optimizer_benchmark/plots/`

- **Logs**: 
  - Segmentation logs: `results/segmentation_output/segmentation.log`
  - Benchmark logs: `results/optimizer_benchmark/logs/`
  - Full workflow log: `results/logs/full_benchmark_[TIMESTAMP].log`

## Key Features

### Parallel Segmentation

The segmentation script uses Python's `ProcessPoolExecutor` to parallelize image processing:

```python
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    # Submit both 10x and 40x tasks concurrently
    futures_10x = [executor.submit(seg_10x_func, img_file) for img_file in image_10x_files]
    futures_40x = [executor.submit(seg_40x_func, img_file) for img_file in image_40x_files]
```

This approach:
- Processes all images (both 10x and 40x) simultaneously
- Maximizes CPU utilization
- Automatically scales to the available cores (leaving one free for system processes)

### Optimizer Benchmark

The benchmark script compares two optimization algorithms:

1. **L-BFGS-B** (Limited-memory Broyden–Fletcher–Goldfarb–Shanno with Bounds)
   - Quasi-Newton method
   - Uses approximation of the Hessian matrix
   - Handles large number of variables efficiently

2. **LMA** (Levenberg-Marquardt Algorithm)
   - Combines Gauss-Newton and gradient descent
   - Adaptive damping parameter
   - Well-suited for non-linear least squares problems

The benchmark measures:
- **RMSD** (Root Mean Square Deviation): Accuracy of the transformation
- **Time**: Computation time in milliseconds
- **Memory**: Peak memory usage in KB
- **Iterations**: Number of iterations until convergence

## Customization

You can customize the benchmark by modifying the script parameters:

- **Segmentation**:
  - Data directory: `--data-dir`
  - Output directory: `--output-dir`
  - Number of workers: `--workers`

- **Benchmark**:
  - Centroids directory: `--centroids-dir`
  - Output directory: `--output-dir`
  - Number of trials: `--trials`
  - Maximum iterations: `--iterations`

For example:

```bash
./run_optimizer_benchmark.sh --trials 10 --iterations 500
```

## System Requirements

- Multiple CPU cores (recommended: 4+ cores)
- Python 3.6+
- NumPy, SciPy, Pandas, Matplotlib
- ImageAnalysis package

## Troubleshooting

If the segmentation phase fails due to missing dependencies like the nd2reader library, the script will fall back to creating mock segmentation results. This allows the workflow to continue for testing purposes.

If you encounter architecture compatibility issues (e.g., Apple Silicon vs. x86_64), consider:
1. Using Rosetta 2 translation for Intel-compiled libraries
2. Creating a conda environment with the appropriate architecture
3. Rebuilding problematic dependencies from source