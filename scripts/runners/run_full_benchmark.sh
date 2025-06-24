#!/bin/bash
# Master script to run the full benchmark process:
# 1. Run parallel segmentation on all images
# 2. Run optimizer benchmark on the resulting nuclear centroids

echo "Starting full benchmark process at $(date)"
echo "CPU Cores available: $(sysctl -n hw.ncpu)"

# Set environment variables for optimal performance
export OPENBLAS_NUM_THREADS=$(sysctl -n hw.ncpu)
export MKL_NUM_THREADS=$(sysctl -n hw.ncpu)
export OMP_NUM_THREADS=$(sysctl -n hw.ncpu)
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
ulimit -n 4096

# Create log directory
mkdir -p results/logs

# Create master log file
MASTER_LOG="results/logs/full_benchmark_$(date +%Y%m%d_%H%M%S).log"

# Print header
echo "===== FULL BENCHMARK PROCESS =====" | tee -a $MASTER_LOG
echo "System: $(sysctl -n machdep.cpu.brand_string)" | tee -a $MASTER_LOG
echo "CPU Cores: $(sysctl -n hw.ncpu)" | tee -a $MASTER_LOG
echo "Memory: $(sysctl -n hw.memsize | awk '{print $0/1024/1024/1024 " GB"}' )" | tee -a $MASTER_LOG
echo "Date: $(date)" | tee -a $MASTER_LOG
echo "===================================" | tee -a $MASTER_LOG
echo "" | tee -a $MASTER_LOG

# STEP 1: Run Parallel Segmentation
echo "STEP 1: Running Parallel Segmentation" | tee -a $MASTER_LOG
echo "Starting at $(date)" | tee -a $MASTER_LOG
echo "" | tee -a $MASTER_LOG

./run_segmentation_parallel.sh | tee -a $MASTER_LOG

SEGMENTATION_RESULT=$?
if [ $SEGMENTATION_RESULT -ne 0 ]; then
  echo "ERROR: Segmentation failed with exit code $SEGMENTATION_RESULT" | tee -a $MASTER_LOG
  echo "Aborting benchmark process." | tee -a $MASTER_LOG
  exit 1
fi

echo "" | tee -a $MASTER_LOG
echo "Segmentation completed at $(date)" | tee -a $MASTER_LOG
echo "" | tee -a $MASTER_LOG

# STEP 2: Run Optimizer Benchmark on Centroids
echo "STEP 2: Running Optimizer Benchmark" | tee -a $MASTER_LOG
echo "Starting at $(date)" | tee -a $MASTER_LOG
echo "" | tee -a $MASTER_LOG

./run_optimizer_benchmark.sh | tee -a $MASTER_LOG

BENCHMARK_RESULT=$?
if [ $BENCHMARK_RESULT -ne 0 ]; then
  echo "ERROR: Optimizer benchmark failed with exit code $BENCHMARK_RESULT" | tee -a $MASTER_LOG
  echo "Benchmark process incomplete." | tee -a $MASTER_LOG
  exit 1
fi

echo "" | tee -a $MASTER_LOG
echo "Optimizer benchmark completed at $(date)" | tee -a $MASTER_LOG
echo "" | tee -a $MASTER_LOG

# FINAL SUMMARY
echo "===== BENCHMARK PROCESS COMPLETE =====" | tee -a $MASTER_LOG
echo "Started: $(head -n 4 $MASTER_LOG | tail -n 1 | cut -d: -f2-)" | tee -a $MASTER_LOG
echo "Finished: $(date)" | tee -a $MASTER_LOG
echo "" | tee -a $MASTER_LOG
echo "Results:" | tee -a $MASTER_LOG
echo "- Segmentation output: results/segmentation_output/" | tee -a $MASTER_LOG
echo "- Optimizer benchmark: results/optimizer_benchmark/" | tee -a $MASTER_LOG
echo "- Full logs: $MASTER_LOG" | tee -a $MASTER_LOG
echo "" | tee -a $MASTER_LOG
echo "To view the optimization benchmark results summary:" | tee -a $MASTER_LOG
echo "cat results/optimizer_benchmark/optimizer_benchmark_overall.csv" | tee -a $MASTER_LOG
echo "====================================" | tee -a $MASTER_LOG