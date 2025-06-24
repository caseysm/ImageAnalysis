# Optimization Algorithm Benchmark

This directory contains benchmark scripts for comparing the L-BFGS-B and Levenberg-Marquardt optimization algorithms for point set registration.

## Architecture Issue Resolution

When trying to run the real data benchmark script, we encountered an architecture compatibility issue:

```
OSError: dlopen(...nd2sdk.framework/Versions/1/nd2sdk, 0x0006): tried: '...' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64e' or 'arm64'))
```

This error occurs because the ND2SDK library used by nd2reader is compiled for x86_64 architecture, but your Apple M3 Pro uses arm64 architecture.

## Solution: Synthetic Data Benchmark

Instead of using real microscopy data that requires the nd2reader library, we've created a synthetic data benchmark that:

1. Generates realistic nuclear-like coordinates 
2. Applies known transformations with controllable noise levels
3. Evaluates both optimization algorithms under identical conditions
4. Uses the same parallelism optimizations as the real data benchmark

This approach allows us to test the parallelism implementation and compare the optimization algorithms without architecture compatibility issues.

## Running the Synthetic Benchmark

```bash
./run_benchmark_synthetic_parallel.sh
```

This script:
- Uses all available CPU cores for parallel processing
- Generates synthetic nuclear coordinates resembling real microscopy data
- Tests different point counts and noise levels
- Compares L-BFGS-B and LMA optimization performance
- Produces comprehensive benchmark results and visualizations

## Benchmark Parameters

The synthetic benchmark can be customized with various parameters:

- `--trials`: Number of repetitions for each configuration (default: 5)
- `--points`: Comma-separated list of point counts to test (default: 100,500,1000)
- `--noise`: Comma-separated list of noise levels to test (default: 0,1,2,5,10)
- `--iterations`: Maximum optimizer iterations (default: 100)

For quick testing, the default script uses a reduced parameter set. For a full benchmark, see the commented section at the end of the script.

## Results

Benchmark results are saved to:

- CSV data: `results/optimizer_benchmark_synthetic/optimizer_benchmark_synthetic.csv`
- Summary: `results/optimizer_benchmark_synthetic/optimizer_benchmark_synthetic_summary.csv`
- Plots: `results/optimizer_benchmark_synthetic/plots/`
- Logs: `results/optimizer_benchmark_synthetic/logs/benchmark_[TIMESTAMP].log`

The results include RMSD, parameter error, computation time, memory usage, and iteration counts for both optimization algorithms across different point counts and noise levels.

## Real Data Solution

To run the real data benchmark on Apple Silicon (M1/M2/M3), you would need:

1. A version of nd2reader/pims_nd2 compiled for arm64 architecture
2. Or use Rosetta 2 translation by creating an x86_64 conda environment

For now, the synthetic benchmark provides a complete platform for evaluating the optimization algorithms with full parallelism.