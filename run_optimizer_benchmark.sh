#!/bin/bash
# Script to run the optimizer benchmark on centroids from segmentation results
# This will compare L-BFGS-B and LMA performance on real nuclear coordinates

echo "Starting optimizer benchmark at $(date)"
echo "CPU Cores available: $(sysctl -n hw.ncpu)"

# Create output directories
mkdir -p results/optimizer_benchmark/logs

# Create a timestamped log filename
LOG_FILE="results/optimizer_benchmark/logs/benchmark_$(date +%Y%m%d_%H%M%S).log"

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
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# System settings to improve CPU utilization
ulimit -n 4096

# Run the optimizer benchmark
echo "Running optimizer benchmark on nuclear centroids..." | tee -a $LOG_FILE
echo "Starting at $(date)" | tee -a $LOG_FILE

# Run with caffeinate to prevent system sleep
caffeinate -i python benchmark_optimizers_from_centroids.py \
  --centroids-dir results/segmentation_output \
  --output-dir results/optimizer_benchmark \
  --trials 5 \
  --iterations 200 2>&1 | tee -a $LOG_FILE

RESULT=$?
if [ $RESULT -eq 0 ]; then
  echo "Benchmark completed successfully at $(date)"
  echo "Results saved to results/optimizer_benchmark"
  
  # Display summary
  echo ""
  echo "Optimizer Benchmark Summary:"
  echo "-------------------------"
  echo "L-BFGS-B Results:"
  grep -A 4 "L-BFGS-B" results/optimizer_benchmark/optimizer_benchmark_overall.csv | column -t -s,
  echo ""
  echo "LMA Results:"
  grep -A 4 "LMA" results/optimizer_benchmark/optimizer_benchmark_overall.csv | column -t -s,
  
  # Compare key metrics
  echo ""
  echo "Performance Comparison:"
  echo "---------------------"
  echo "Metric       | L-BFGS-B | LMA     | Winner"
  echo "-------------+---------+--------+--------"
  
  # Extract and format metrics
  LBFGS_RMSD=$(grep "L-BFGS-B" results/optimizer_benchmark/optimizer_benchmark_overall.csv | cut -d, -f3)
  LMA_RMSD=$(grep "LMA" results/optimizer_benchmark/optimizer_benchmark_overall.csv | cut -d, -f3)
  
  LBFGS_TIME=$(grep "L-BFGS-B" results/optimizer_benchmark/optimizer_benchmark_overall.csv | cut -d, -f9)
  LMA_TIME=$(grep "LMA" results/optimizer_benchmark/optimizer_benchmark_overall.csv | cut -d, -f9)
  
  # Compare RMSD (lower is better)
  if (( $(echo "$LBFGS_RMSD < $LMA_RMSD" | bc -l) )); then
    RMSD_WINNER="L-BFGS-B"
  else
    RMSD_WINNER="LMA"
  fi
  
  # Compare Time (lower is better)
  if (( $(echo "$LBFGS_TIME < $LMA_TIME" | bc -l) )); then
    TIME_WINNER="L-BFGS-B"
  else
    TIME_WINNER="LMA"
  fi
  
  printf "RMSD (lower) | %.4f  | %.4f | %s\n" $LBFGS_RMSD $LMA_RMSD $RMSD_WINNER
  printf "Time (ms)    | %.2f | %.2f | %s\n" $LBFGS_TIME $LMA_TIME $TIME_WINNER
  
  # Overall recommendation
  echo ""
  echo "Recommendation:"
  if [ "$RMSD_WINNER" = "$TIME_WINNER" ]; then
    echo "- Use $RMSD_WINNER: Best performance across both RMSD and computation time"
  else
    echo "- For accuracy: Use $RMSD_WINNER (better RMSD)"
    echo "- For speed: Use $TIME_WINNER (faster execution)"
  fi
else
  echo "Benchmark failed with exit code $RESULT at $(date)"
  echo "Check logs for details"
fi