# Parallelized Benchmark Implementation Summary

## Architecture Compatibility Solution

We encountered an architecture compatibility issue when running the real data benchmark, as the `nd2reader` library's ND2SDK dependencies are compiled for x86_64 architecture but not arm64 (Apple Silicon). To address this, we implemented a fully-functional synthetic data benchmark that:

1. Generates realistic nuclear-like coordinates resembling real microscopy data
2. Applies controlled transformations with adjustable noise levels
3. Implements identical parallelism optimizations
4. Allows for comprehensive benchmarking of the optimization algorithms

## Parallelism Implementation

Both benchmarks (synthetic and real data) implement multiple levels of parallelism:

### 1. Parallel Segmentation (Real Data Version)
```python
# Process 10x and 40x images in parallel with ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=num_cores) as executor:
    # Submit both 10x and 40x tasks in parallel
    futures_10x = [executor.submit(seg_10x_partial, img_file) for img_file in image_10x_files]
    futures_40x = [executor.submit(seg_40x_partial, img_file) for img_file in image_40x_files]
```

### 2. Parallel Benchmark Configurations (Synthetic Version)
```python
# Run benchmarks in parallel
with ProcessPoolExecutor(max_workers=num_cores) as executor:
    # Submit all benchmark configurations
    futures = [executor.submit(run_single_benchmark, *config) for config in benchmark_configs]
    
    # Collect results as they complete
    for i, future in enumerate(futures):
        trial_results = future.result()
        results.extend(trial_results)
```

### 3. Parallel Trial Execution (Both Versions)
```python
# Run trials in parallel
with ProcessPoolExecutor(max_workers=min(num_cores, num_trials)) as executor:
    # Submit all trials for parallel execution
    futures = [executor.submit(run_optimizer_trial, trial) for trial in range(num_trials)]
```

### 4. Environment Optimizations (Both Versions)
```bash
# Set numerical libraries to use all available cores
export OPENBLAS_NUM_THREADS=$(sysctl -n hw.ncpu)
export MKL_NUM_THREADS=$(sysctl -n hw.ncpu)
export OMP_NUM_THREADS=$(sysctl -n hw.ncpu)

# Enable Python to use multiprocessing effectively
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
```

## Performance Monitoring

Both benchmark implementations include real-time monitoring of CPU utilization:

```bash
# Display process information every 30 seconds
while ps -p $BENCHMARK_PID > /dev/null; do
    echo "$(date +%H:%M:%S) - CPU Usage:" | tee -a $LOG_FILE
    ps -p $BENCHMARK_PID -o %cpu,%mem | tee -a $LOG_FILE
    echo "System-wide CPU load:" | tee -a $LOG_FILE
    top -l 1 -n 1 | grep "CPU usage" | tee -a $LOG_FILE
    echo "---" | tee -a $LOG_FILE
    sleep 30
done
```

## Usage Instructions

### Synthetic Data Benchmark (Recommended)
```bash
./run_benchmark_synthetic_parallel.sh
```

### Real Data Benchmark (Requires x86_64 compatibility)
```bash
./run_benchmark_realdata.sh
```

## Key Benefits of the Implementation

1. **Full Core Utilization**: Both benchmarks are designed to utilize all available CPU cores
2. **Configurable Parameters**: Easily adjustable point counts, noise levels, and trial counts
3. **Comprehensive Metrics**: Measures RMSD, parameter error, computation time, memory usage, and iterations
4. **Detailed Visualization**: Generates plots comparing optimizers across different conditions
5. **Real-time Monitoring**: Provides continuous feedback on CPU utilization during execution
6. **Architecture Independence**: Synthetic benchmark works on any architecture without dependencies

## Next Steps

1. Run the synthetic benchmark to get comparative performance of L-BFGS-B and LMA
2. Analyze the results to determine which optimizer is better suited for nuclear point registration
3. Apply the findings to improve the mapping component of the image analysis pipeline