# Running the Real Data Optimizer Benchmark

This document explains how to run the real data benchmark for comparing L-BFGS-B and LMA optimization algorithms on your actual microscopy data.

## Background

The benchmark processes your actual 10x and 40x microscopy images through the full image analysis pipeline, extracting real nuclear coordinates to test the optimizers. This process takes approximately 30+ minutes to complete, as it includes:

1. Segmenting microscopy images (most time-consuming step)
2. Extracting nuclear centroids
3. Finding corresponding points
4. Running both optimizers multiple times

## Running the Benchmark

### Method 1: Using the Shell Script

We've provided a shell script that runs the benchmark and captures the output to a log file:

```bash
# From the project root directory
./run_benchmark_realdata.sh
```

This script will:
- Create necessary output directories
- Run the benchmark with standard settings (3 trials)
- Save output to a timestamped log file

### Method 2: Manual Execution

You can also run the benchmark manually with custom parameters:

```bash
# Run with default settings
python benchmark_optimizers_real_data.py

# Run with custom settings
python benchmark_optimizers_real_data.py \
  --trials 5 \
  --output-dir custom_results/real_benchmark
```

### Running in a Terminal Session

For long-running processes, consider using one of these approaches:

1. **Screen/tmux sessions** (if available):
   ```bash
   # Start a screen session
   screen -S benchmark
   
   # Run the benchmark
   ./run_benchmark_realdata.sh
   
   # Detach from session (Ctrl+A, then D)
   # Later, reattach with:
   screen -r benchmark
   ```

2. **Nohup** (to prevent termination when terminal closes):
   ```bash
   nohup ./run_benchmark_realdata.sh &
   ```

## Viewing Results

After the benchmark completes, results will be available in the output directory:

```
results/optimizer_benchmark_real_data/
├── optimizer_benchmark_real_data.csv       # Raw benchmark data
├── optimizer_benchmark_real_data_summary.csv  # Statistical summary
└── plots/
    ├── rmsd_comparison_real_data.png      # RMSD comparison plot
    └── time_comparison_real_data.png      # Time comparison plot
```

The summary CSV contains the key metrics averaged across all trials, which you can use to compare the performance of L-BFGS-B and LMA on your real data.

## Interpreting Results

When analyzing the results:

1. **RMSD values**: Lower values indicate better alignment between 10x and 40x coordinates
2. **Computation time**: Typically measured in milliseconds
3. **Optimizer parameters**: Check if the recovered parameters (a, b, c, d, tx, ty) align with expected values (~4.0 scale factor for 10x→40x)

Note that real data results may show different characteristics than synthetic data due to actual biological variation and microscopy artifacts.