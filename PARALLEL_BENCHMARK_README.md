# Parallel Benchmark Implementation

This document describes the parallelism implementation for the optimization benchmark scripts.

## Overview

The benchmark has been enhanced to utilize all available CPU cores through parallel processing in several key areas:

1. **Parallel Segmentation**: Both 10x and 40x image segmentation tasks run concurrently
2. **Parallel Trials**: Multiple benchmark trials run in parallel
3. **Optimized Library Settings**: Numerical libraries configured to utilize all cores
4. **Resource Monitoring**: Real-time CPU utilization tracking

## Implementation Details

### Segmentation Parallelism

Both 10x and 40x image segmentation tasks are now submitted concurrently to a `ProcessPoolExecutor`. 
This ensures maximum core utilization rather than processing 10x and 40x images sequentially.

```python
# Submit both 10x and 40x tasks in parallel to maximize core usage
futures_10x = [executor.submit(seg_10x_partial, img_file) for img_file in image_10x_files]
futures_40x = [executor.submit(seg_40x_partial, img_file) for img_file in image_40x_files]
```

### Benchmark Trial Parallelism

Each optimization benchmark trial (comparing L-BFGS-B and LMA) now runs in parallel:

```python
# Run trials in parallel
with ProcessPoolExecutor(max_workers=min(num_cores, num_trials)) as executor:
    # Submit all trials for parallel execution
    futures = [executor.submit(run_optimizer_trial, trial) for trial in range(num_trials)]
```

### Environment Optimizations

The shell script configures environment variables to maximize parallel performance:

```bash
# Set numerical libraries to use all available cores
export OPENBLAS_NUM_THREADS=$(sysctl -n hw.ncpu)
export MKL_NUM_THREADS=$(sysctl -n hw.ncpu)
export OMP_NUM_THREADS=$(sysctl -n hw.ncpu)

# Enable Python to use multiprocessing effectively
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# System settings to improve CPU utilization
ulimit -n 4096
```

### Performance Monitoring

Real-time monitoring is implemented to track CPU utilization:

```bash
# Display process information every 30 seconds
while ps -p $BENCHMARK_PID > /dev/null; do
    echo "$(date +%H:%M:%S) - CPU Usage:" | tee -a $LOG_FILE
    ps -p $BENCHMARK_PID -o %cpu,%mem | tee -a $LOG_FILE
    # ...additional monitoring...
    sleep 30
done
```

## Running the Benchmark

To execute the parallelized benchmark:

```bash
# Make the script executable
chmod +x run_benchmark_realdata.sh

# Run the benchmark
./run_benchmark_realdata.sh
```

The script will automatically utilize all available CPU cores and provide real-time monitoring of resource usage.

## Expected Performance

The parallelization should provide significant speedup on multi-core systems:

- **Linear scaling** with number of cores for the segmentation phase
- **Near-linear scaling** for the trial phase (if running multiple trials)
- **Overall speedup** depends on the number of available cores and the balance between segmentation and optimization workloads

## Logs and Results

Results are saved to:
- CSV data: `results/optimizer_benchmark_real_data/optimizer_benchmark_real_data.csv`
- Summary: `results/optimizer_benchmark_real_data/optimizer_benchmark_real_data_summary.csv`
- Plots: `results/optimizer_benchmark_real_data/plots/`
- Logs: `results/optimizer_benchmark_real_data/logs/benchmark_[TIMESTAMP].log`

The logs contain detailed information about CPU utilization throughout the benchmark run.