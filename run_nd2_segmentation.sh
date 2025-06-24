#!/bin/bash
# Run segmentation on ND2 files in parallel

# Get the number of available cores (leave one free for system processes)
NUM_CORES=$(($(sysctl -n hw.ncpu) - 1))
echo "Running with $NUM_CORES cores"

# Set base directories
DATA_DIR="/Users/csm70/Desktop/Shalem_Lab/ImageAnalysis/Data"
OUTPUT_DIR="/Users/csm70/Desktop/Shalem_Lab/ImageAnalysis/results/nd2_segmentation_output"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Optional limit on number of files to process
LIMIT=${1:-5}  # Default to 5 files if not specified
echo "Processing up to $LIMIT ND2 files"

# Run the parallel segmentation
python run_parallel_nd2_segmentation.py \
    --input "$DATA_DIR" \
    --output "$OUTPUT_DIR" \
    --cores "$NUM_CORES" \
    --limit "$LIMIT"

echo "Segmentation completed. Results saved to $OUTPUT_DIR"

# Run the optimizer benchmark on the segmentation results
echo "Running optimizer benchmark on segmentation results..."
python benchmark_optimizers_from_centroids.py \
    --centroids-dir "$OUTPUT_DIR" \
    --output-dir "results/nd2_optimizer_benchmark" \
    --trials 5

echo "Benchmark completed. Results saved to results/nd2_optimizer_benchmark"