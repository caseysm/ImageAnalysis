#!/bin/bash
# Run 10X and 40X segmentation and generate optimal transformation matrix

# Get the number of available cores (leave one free for system processes)
NUM_CORES=$(($(sysctl -n hw.ncpu) - 1))
echo "Running with $NUM_CORES cores"

# Set base directories
DATA_DIR="/Users/csm70/Desktop/Shalem_Lab/ImageAnalysis/Data"
OUTPUT_DIR="/Users/csm70/Desktop/Shalem_Lab/ImageAnalysis/results/nd2_segmentation_combined"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# First process 10X files (genotyping/cycle_1)
echo "Processing 10X files from genotyping/cycle_1..."
python run_parallel_nd2_segmentation_v2.py \
    --input "$DATA_DIR/genotyping/cycle_1" \
    --output "$OUTPUT_DIR" \
    --cores "$NUM_CORES" \
    --limit 2  # Process 2 files to keep runtime manageable

# Next process 40X files (phenotyping)
echo "Processing 40X files from phenotyping..."
python run_parallel_nd2_segmentation_v2.py \
    --input "$DATA_DIR/phenotyping" \
    --output "$OUTPUT_DIR" \
    --cores "$NUM_CORES" \
    --limit 2  # Process 2 files to keep runtime manageable

echo "Segmentation completed. Results saved to $OUTPUT_DIR"

# Run the optimizer benchmark to compare methods and generate transformation matrix
echo "Running optimizer benchmark to generate transformation matrix..."
python benchmark_optimizers_from_centroids.py \
    --centroids-dir "$OUTPUT_DIR" \
    --output-dir "$OUTPUT_DIR/optimizer_results" \
    --trials 5 \
    --iterations 200

echo "Benchmark completed. Results saved to $OUTPUT_DIR/optimizer_results"

# Display results summary
if [ -f "$OUTPUT_DIR/optimizer_results/optimizer_benchmark_overall.csv" ]; then
    echo ""
    echo "Transformation Matrix Optimization Results:"
    echo "-----------------------------------------"
    echo "L-BFGS-B Results:"
    grep -A 4 "L-BFGS-B" "$OUTPUT_DIR/optimizer_results/optimizer_benchmark_overall.csv" | column -t -s,
    echo ""
    echo "LMA Results:"
    grep -A 4 "LMA" "$OUTPUT_DIR/optimizer_results/optimizer_benchmark_overall.csv" | column -t -s,
fi

# Find the best transformation matrix (typically from L-BFGS-B)
echo ""
echo "Best transformation parameters (from L-BFGS-B):"
head -n 2 "$OUTPUT_DIR/optimizer_results/optimizer_benchmark_real_points.csv" | column -t -s,
grep "L-BFGS-B" "$OUTPUT_DIR/optimizer_results/optimizer_benchmark_real_points.csv" | sort -n -k7 | head -n 1 | column -t -s,