#!/bin/bash
# Batch script to run the synthetic optimizer benchmark with parallel processing
# This will run the benchmark utilizing all available CPU cores

echo "Starting synthetic parallel benchmark at $(date)"
echo "CPU Cores available: $(sysctl -n hw.ncpu)"

# Create output directories
mkdir -p results/optimizer_benchmark_synthetic/logs

# Create a timestamped log filename
LOG_FILE="results/optimizer_benchmark_synthetic/logs/benchmark_$(date +%Y%m%d_%H%M%S).log"

# Print system information
echo "CPU Information:" | tee -a $LOG_FILE
sysctl -n machdep.cpu.brand_string | tee -a $LOG_FILE
echo "Number of CPU cores: $(sysctl -n hw.ncpu)" | tee -a $LOG_FILE
echo "Memory: $(sysctl -n hw.memsize | awk '{print $0/1024/1024/1024 " GB"}' )" | tee -a $LOG_FILE

# Set numerical libraries to use all available cores
export OPENBLAS_NUM_THREADS=$(sysctl -n hw.ncpu)
export MKL_NUM_THREADS=$(sysctl -n hw.ncpu)
export OMP_NUM_THREADS=$(sysctl -n hw.ncpu)

# Enable Python to use multiprocessing effectively
export PYTHONUNBUFFERED=1  # Ensures real-time logging output
export PYTHONFAULTHANDLER=1  # Better error reporting in parallel contexts

# System settings to improve CPU utilization
ulimit -n 4096  # Increase file descriptor limit for parallel processes

# Run the benchmark with parallelized execution
echo "Running benchmark with parallel execution on multiple cores..." | tee -a $LOG_FILE
echo "Starting at $(date)" | tee -a $LOG_FILE

# Run with caffeinate to prevent system sleep and real-time monitoring
# For a quick test, use fewer points and trials
(caffeinate -i python benchmark_optimizers_synthetic_parallel.py \
  --trials 3 \
  --points 100,500 \
  --noise 0,2,5 \
  --output-dir results/optimizer_benchmark_synthetic 2>&1 | tee -a $LOG_FILE) &

# Get the PID of the benchmark process
BENCHMARK_PID=$!

# Display process information every 30 seconds while it's running
echo "Monitoring CPU usage of benchmark process..." | tee -a $LOG_FILE
(
  while ps -p $BENCHMARK_PID > /dev/null; do
    echo "$(date +%H:%M:%S) - CPU Usage:" | tee -a $LOG_FILE
    ps -p $BENCHMARK_PID -o %cpu,%mem | tee -a $LOG_FILE
    echo "System-wide CPU load:" | tee -a $LOG_FILE
    top -l 1 -n 1 | grep "CPU usage" | tee -a $LOG_FILE
    echo "---" | tee -a $LOG_FILE
    sleep 30
  done
) &

# Wait for the benchmark to complete
wait $BENCHMARK_PID
echo "Benchmark completed at $(date)" | tee -a $LOG_FILE

RESULT=$?
if [ $RESULT -eq 0 ]; then
  echo "Benchmark completed successfully at $(date)"
else
  echo "Benchmark failed with exit code $RESULT at $(date)"
  echo "Check logs for details"
fi

# For a full benchmark run, uncomment and use these settings
# python benchmark_optimizers_synthetic_parallel.py \
#   --trials 5 \
#   --points 100,500,1000,2000 \
#   --noise 0,1,2,5,10 \
#   --output-dir results/optimizer_benchmark_synthetic