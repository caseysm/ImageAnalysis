#!/bin/bash
# Run segmentation on TIFF files converted from ND2 files

# Get the number of available cores (leave one free for system processes)
NUM_CORES=$(($(sysctl -n hw.ncpu) - 1))
echo "Running with $NUM_CORES cores"

# Set base directories
DATA_DIR="/Users/csm70/Desktop/Shalem_Lab/ImageAnalysis/Data"
OUTPUT_DIR="/Users/csm70/Desktop/Shalem_Lab/ImageAnalysis/results/tiff_segmentation_output"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Find all converted TIFF files
echo "Looking for TIFF files in $DATA_DIR"

# Create temporary files to store the file lists
NUCLEAR_FILES=$(mktemp)
CELL_FILES=$(mktemp)
MAGNIFICATIONS=$(mktemp)

# Find all nuclear and cell TIFF files
find "$DATA_DIR" -name "*.c0.tiff" > "$NUCLEAR_FILES"  # Assuming c0 is nuclear (DAPI)
find "$DATA_DIR" -name "*.c1.tiff" > "$CELL_FILES"     # Assuming c1 is cell marker

# Determine magnification based on path
while read -r nuclear_file; do
    if [[ "$nuclear_file" == *"/genotyping/"* ]] || [[ "$nuclear_file" == *"10X"* ]]; then
        echo "10X" >> "$MAGNIFICATIONS"
    else
        echo "40X" >> "$MAGNIFICATIONS"
    fi
done < "$NUCLEAR_FILES"

# Count files
NUM_FILES=$(wc -l < "$NUCLEAR_FILES")
echo "Found $NUM_FILES nuclear channel TIFF files"

# Function to process a single file
process_file() {
    local nuclear_file=$1
    local cell_file=$2
    local magnification=$3
    local index=$4
    local total=$5
    
    # Extract basename for logging
    filename=$(basename "$nuclear_file")
    
    echo "[$index/$total] Processing $filename at $magnification magnification"
    
    # Run segmentation
    python tiff_segmentation.py \
        --nuclear "$nuclear_file" \
        --cell "$cell_file" \
        --output "$OUTPUT_DIR" \
        --mag "$magnification"
    
    return $?
}

export -f process_file

# Process files in parallel using GNU Parallel if available
if command -v parallel >/dev/null 2>&1; then
    echo "Using GNU Parallel to process files"
    paste -d " " "$NUCLEAR_FILES" "$CELL_FILES" "$MAGNIFICATIONS" |
        nl -w1 -s " " |
        parallel -j "$NUM_CORES" --colsep ' ' process_file {2} {3} {4} {1} "$NUM_FILES"
else
    # Fall back to sequential processing
    echo "GNU Parallel not found, processing files sequentially"
    
    i=1
    paste -d " " "$NUCLEAR_FILES" "$CELL_FILES" "$MAGNIFICATIONS" | while read -r nuclear_file cell_file magnification; do
        process_file "$nuclear_file" "$cell_file" "$magnification" $i "$NUM_FILES"
        i=$((i+1))
    done
fi

# Clean up temporary files
rm "$NUCLEAR_FILES" "$CELL_FILES" "$MAGNIFICATIONS"

echo "All files processed. Results saved to $OUTPUT_DIR"

# Run the optimizer benchmark on the segmentation results
echo "Running optimizer benchmark on segmentation results..."
python benchmark_optimizers_from_centroids.py \
    --centroids-dir "$OUTPUT_DIR" \
    --output-dir "results/tiff_optimizer_benchmark" \
    --trials 5

echo "Benchmark completed. Results saved to results/tiff_optimizer_benchmark"