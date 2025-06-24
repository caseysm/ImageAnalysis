#!/bin/bash
# Script to benchmark parallel segmentation performance
# This will test segmentation parallelization across different core counts

echo "Starting segmentation parallelization benchmark at $(date)"
echo "CPU Cores available: $(sysctl -n hw.ncpu)"

# Create output directories
mkdir -p results/segmentation_benchmark/logs

# Create a timestamped log filename
LOG_FILE="results/segmentation_benchmark/logs/benchmark_$(date +%Y%m%d_%H%M%S).log"

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

# Run the focused segmentation benchmark
echo "Running segmentation parallelization benchmark..." | tee -a $LOG_FILE
echo "Starting at $(date)" | tee -a $LOG_FILE

# Run benchmark with different image count and complexity levels
# - Adjust the number of images to match your available cores
# - Complexity factor affects the processing time per image
python parallel_segmentation_benchmark.py \
  --images 24 \
  --width 1024 \
  --height 1024 \
  --complexity 3 \
  --output-dir results/segmentation_benchmark 2>&1 | tee -a $LOG_FILE

RESULT=$?
if [ $RESULT -eq 0 ]; then
  echo "Benchmark completed successfully at $(date)"
  
  # Display the recommendation
  echo ""
  echo "SEGMENTATION PARALLELIZATION RECOMMENDATION:"
  echo "--------------------------------------------"
  cat results/segmentation_benchmark/recommendation.txt
else
  echo "Benchmark failed with exit code $RESULT at $(date)"
  echo "Check logs for details"
fi