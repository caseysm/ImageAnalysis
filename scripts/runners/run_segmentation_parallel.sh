#!/bin/bash
# Script to run the segmentation pipeline in parallel
# This will process all 10x and 40x images concurrently using all available cores

echo "Starting parallel segmentation pipeline at $(date)"
echo "CPU Cores available: $(sysctl -n hw.ncpu)"

# Install tqdm if not already installed
if ! python -c "import tqdm" 2>/dev/null; then
    echo "Installing tqdm package for progress tracking..."
    pip install tqdm
fi

# Create output directories
mkdir -p results/segmentation_output

# Set numerical libraries to use all available cores
export OPENBLAS_NUM_THREADS=$(sysctl -n hw.ncpu)
export MKL_NUM_THREADS=$(sysctl -n hw.ncpu)
export OMP_NUM_THREADS=$(sysctl -n hw.ncpu)

# Enable Python to use multiprocessing effectively
export PYTHONUNBUFFERED=1  # Ensures real-time logging output
export PYTHONFAULTHANDLER=1  # Better error reporting in parallel contexts

# System settings to improve CPU utilization
ulimit -n 4096  # Increase file descriptor limit for parallel processes

# Run the segmentation pipeline in parallel
echo "Running segmentation pipeline with parallel processing on all cores..."
echo "Starting at $(date)"

# Display progress monitoring instructions
echo ""
echo "==========================================================="
echo "  PROGRESS MONITORING INSTRUCTIONS"
echo "==========================================================="
echo "  To check segmentation progress, run in a new terminal:"
echo "  ./check_segmentation_progress.sh"
echo ""
echo "  This will show completed images, percentage, and CPU usage"
echo "==========================================================="
echo ""

# Run with caffeinate to prevent system sleep
caffeinate -i python run_parallel_segmentation.py \
  --data-dir /Users/csm70/Desktop/Shalem_Lab/ImageAnalysis/Data \
  --output-dir results/segmentation_output \
  --clean

RESULT=$?
if [ $RESULT -eq 0 ]; then
  echo "Segmentation completed successfully at $(date)"
  echo "Results saved to results/segmentation_output"
  
  # Display basic statistics
  echo ""
  echo "Segmentation Results Summary:"
  echo "--------------------------"
  cat results/segmentation_output/segmentation_summary.json
  
  # Now that segmentation is complete, set up for running the benchmark on the extracted centroids
  echo ""
  echo "To run the optimizer benchmark on the extracted centroids, use:"
  echo "python benchmark_optimizers.py --centroids-dir results/segmentation_output"
else
  echo "Segmentation failed with exit code $RESULT at $(date)"
  echo "Check logs for details: results/segmentation_output/segmentation.log"
fi