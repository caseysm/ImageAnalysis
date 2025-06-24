#!/bin/bash
# Script to run the large-scale optimizer benchmark comparing L-BFGS-B and LMA algorithms
# on synthetic datasets ranging from 100 to 100,000 cells

echo "Starting large-scale optimizer benchmark at $(date)"
echo "CPU Cores available: $(sysctl -n hw.ncpu)"

# Create output directories
RESULTS_DIR="results/optimizer_benchmark_large_scale"
mkdir -p $RESULTS_DIR/logs

# Create a timestamped log filename
LOG_FILE="$RESULTS_DIR/logs/benchmark_$(date +%Y%m%d_%H%M%S).log"

# Print system information
echo "===== SYSTEM INFORMATION =====" | tee -a $LOG_FILE
echo "CPU Information:" | tee -a $LOG_FILE
sysctl -n machdep.cpu.brand_string | tee -a $LOG_FILE
echo "Number of CPU cores: $(sysctl -n hw.ncpu)" | tee -a $LOG_FILE
echo "Memory: $(sysctl -n hw.memsize | awk '{print $0/1024/1024/1024 " GB"}' )" | tee -a $LOG_FILE
echo "============================" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Set numerical libraries to use all available cores
export OPENBLAS_NUM_THREADS=$(sysctl -n hw.ncpu)
export MKL_NUM_THREADS=$(sysctl -n hw.ncpu)
export OMP_NUM_THREADS=$(sysctl -n hw.ncpu)

# Enable Python to use multiprocessing effectively
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# System settings to improve CPU utilization
ulimit -n 4096

# Available run modes
function show_usage {
    echo "Usage: $0 [--mode MODE] [--no-microscopy-sim]"
    echo ""
    echo "Available modes:"
    echo "  quick       - Run a quick test with small point counts (100-1000)"
    echo "  medium      - Run a medium test with moderate point counts (100-10,000)"
    echo "  full        - Run the full benchmark with all point counts (100-100,000)"
    echo "  noise       - Run a noise sensitivity test with varied noise levels"
    echo "  outliers    - Run an outlier sensitivity test with varied outlier fractions"
    echo "  custom      - Run with custom parameters (edit script to modify)"
    echo ""
    echo "Options:"
    echo "  --no-microscopy-sim  - Disable realistic 10x/40x microscopy simulation"
    echo ""
    echo "Default mode: medium"
    exit 1
}

# Parse command line arguments
MODE="medium"
MICROSCOPY_SIM="--microscopy-simulation"

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --no-microscopy-sim)
            MICROSCOPY_SIM=""
            shift
            ;;
        --help|-h)
            show_usage
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            ;;
    esac
done

echo "Selected run mode: $MODE" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Define benchmark parameters based on selected mode
case $MODE in
    quick)
        POINT_COUNTS="100,500,1000"
        NOISE_LEVELS="0.01,0.1"
        OUTLIER_FRACTIONS="0,0.1"
        TRIALS=2
        MAX_ITER=50
        echo "Running quick benchmark (small point counts, fewer trials)" | tee -a $LOG_FILE
        ;;
        
    medium)
        POINT_COUNTS="100,500,1000,5000,10000"
        NOISE_LEVELS="0.01,0.05,0.1"
        OUTLIER_FRACTIONS="0,0.1,0.2"
        TRIALS=3
        MAX_ITER=100
        echo "Running medium benchmark (moderate point counts)" | tee -a $LOG_FILE
        ;;
        
    full)
        POINT_COUNTS="100,500,1000,5000,10000,50000,100000"
        NOISE_LEVELS="0.01,0.05,0.1"
        OUTLIER_FRACTIONS="0,0.1,0.2"
        TRIALS=5
        MAX_ITER=200
        echo "Running full benchmark (all point counts)" | tee -a $LOG_FILE
        ;;
        
    noise)
        POINT_COUNTS="100,1000,10000"
        NOISE_LEVELS="0.01,0.025,0.05,0.075,0.1,0.15,0.2"
        OUTLIER_FRACTIONS="0,0.1"
        TRIALS=3
        MAX_ITER=100
        echo "Running noise sensitivity benchmark (varied noise levels)" | tee -a $LOG_FILE
        ;;
        
    outliers)
        POINT_COUNTS="100,1000,10000"
        NOISE_LEVELS="0.01,0.1"
        OUTLIER_FRACTIONS="0,0.05,0.1,0.15,0.2,0.25,0.3"
        TRIALS=3
        MAX_ITER=100
        echo "Running outlier sensitivity benchmark (varied outlier fractions)" | tee -a $LOG_FILE
        ;;
        
    custom)
        # Custom parameters - modify these as needed
        POINT_COUNTS="100,1000,10000,100000"
        NOISE_LEVELS="0.01,0.1,0.2"
        OUTLIER_FRACTIONS="0,0.1,0.2"
        TRIALS=3
        MAX_ITER=100
        echo "Running custom benchmark with user-defined parameters" | tee -a $LOG_FILE
        ;;
        
    *)
        echo "Invalid mode: $MODE"
        show_usage
        ;;
esac

# Print configuration
echo "Benchmark Configuration:" | tee -a $LOG_FILE
echo "- Point counts: $POINT_COUNTS" | tee -a $LOG_FILE
echo "- Noise levels: $NOISE_LEVELS" | tee -a $LOG_FILE
echo "- Outlier fractions: $OUTLIER_FRACTIONS" | tee -a $LOG_FILE
echo "- Trials per configuration: $TRIALS" | tee -a $LOG_FILE
echo "- Maximum iterations: $MAX_ITER" | tee -a $LOG_FILE
echo "- Results directory: $RESULTS_DIR" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Run the benchmark with caffeinate to prevent system sleep
echo "Running optimizer benchmark..." | tee -a $LOG_FILE
echo "Starting at $(date)" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Run with caffeinate to prevent system sleep
(caffeinate -i python benchmark_optimizers_large_scale.py \
  --point-counts $POINT_COUNTS \
  --noise-levels $NOISE_LEVELS \
  --outlier-fractions $OUTLIER_FRACTIONS \
  --trials $TRIALS \
  --max-iter $MAX_ITER \
  --results-dir $RESULTS_DIR \
  $MICROSCOPY_SIM 2>&1 | tee -a $LOG_FILE) &

# Get the PID of the benchmark process
BENCHMARK_PID=$!

# Display process information every minute while it's running
echo "Monitoring CPU usage during benchmark (updated every minute)..." | tee -a $LOG_FILE
(
  while ps -p $BENCHMARK_PID > /dev/null; do
    echo "$(date +%H:%M:%S) - CPU & Memory Usage:" | tee -a $LOG_FILE
    ps -p $BENCHMARK_PID -o %cpu,%mem,rss | tee -a $LOG_FILE
    echo "---" | tee -a $LOG_FILE
    sleep 60
  done
) &

# Wait for the benchmark to complete
wait $BENCHMARK_PID
RESULT=$?

if [ $RESULT -eq 0 ]; then
  echo "" | tee -a $LOG_FILE
  echo "Benchmark completed successfully at $(date)" | tee -a $LOG_FILE
  echo "Results saved to $RESULTS_DIR" | tee -a $LOG_FILE

  # Run additional visualizations
  echo "" | tee -a $LOG_FILE
  echo "Generating additional visualizations..." | tee -a $LOG_FILE
  python add_extra_visualizations.py --results-dir $RESULTS_DIR | tee -a $LOG_FILE
  echo "Advanced visualizations completed" | tee -a $LOG_FILE
  
  # Display summary if available
  SUMMARY_FILE="$RESULTS_DIR/large_scale_benchmark_summary.csv"
  if [ -f "$SUMMARY_FILE" ]; then
    echo "" | tee -a $LOG_FILE
    echo "Benchmark Summary:" | tee -a $LOG_FILE
    echo "===================" | tee -a $LOG_FILE
    
    # L-BFGS-B average results
    echo "L-BFGS-B Performance:" | tee -a $LOG_FILE
    echo "-------------------" | tee -a $LOG_FILE
    grep "L-BFGS-B" $SUMMARY_FILE | awk -F, '{sum_time+=$5; sum_rmsd+=$3; sum_mem+=$7; count++} END {printf "Average Time: %.2f ms\nAverage RMSD: %.6f\nAverage Memory: %.2f KB\n", sum_time/count, sum_rmsd/count, sum_mem/count}' | tee -a $LOG_FILE
    
    # LMA average results
    echo "" | tee -a $LOG_FILE
    echo "LMA Performance:" | tee -a $LOG_FILE
    echo "---------------" | tee -a $LOG_FILE
    grep "LMA" $SUMMARY_FILE | awk -F, '{sum_time+=$5; sum_rmsd+=$3; sum_mem+=$7; count++} END {printf "Average Time: %.2f ms\nAverage RMSD: %.6f\nAverage Memory: %.2f KB\n", sum_time/count, sum_rmsd/count, sum_mem/count}' | tee -a $LOG_FILE
    
    # Performance comparison for large datasets
    echo "" | tee -a $LOG_FILE
    echo "Performance on Large Datasets (10,000+ points):" | tee -a $LOG_FILE
    echo "-------------------------------------------" | tee -a $LOG_FILE
    grep "L-BFGS-B" $SUMMARY_FILE | awk -F, '{if ($2 >= 10000) {sum_time+=$5; sum_rmsd+=$3; count++}} END {if (count > 0) printf "L-BFGS-B: Avg Time: %.2f ms, Avg RMSD: %.6f (across %d large datasets)\n", sum_time/count, sum_rmsd/count, count; else print "No large datasets found for L-BFGS-B"}' | tee -a $LOG_FILE
    grep "LMA" $SUMMARY_FILE | awk -F, '{if ($2 >= 10000) {sum_time+=$5; sum_rmsd+=$3; count++}} END {if (count > 0) printf "LMA: Avg Time: %.2f ms, Avg RMSD: %.6f (across %d large datasets)\n", sum_time/count, sum_rmsd/count, count; else print "No large datasets found for LMA"}' | tee -a $LOG_FILE
  fi
  
  # Provide instructions for viewing results
  echo "" | tee -a $LOG_FILE
  echo "To view visualization plots, check: $RESULTS_DIR/plots/" | tee -a $LOG_FILE
  echo "To view detailed data, check: $RESULTS_DIR/large_scale_benchmark.csv" | tee -a $LOG_FILE
else
  echo "" | tee -a $LOG_FILE
  echo "Benchmark failed with exit code $RESULT at $(date)" | tee -a $LOG_FILE
  echo "Check logs for details: $LOG_FILE" | tee -a $LOG_FILE
fi