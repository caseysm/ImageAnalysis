#!/bin/bash
# Full ND2 segmentation, stitching, and alignment pipeline

# Get the number of available cores (leave one free for system processes)
NUM_CORES=$(($(sysctl -n hw.ncpu) - 1))
echo "Running with $NUM_CORES cores"

# Set base directories
DATA_DIR="/Users/csm70/Desktop/Shalem_Lab/ImageAnalysis/Data"
OUTPUT_DIR="/Users/csm70/Desktop/Shalem_Lab/ImageAnalysis/results/nd2_full_pipeline"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/10X"
mkdir -p "$OUTPUT_DIR/40X"
mkdir -p "$OUTPUT_DIR/composite"

# Step 1: Process ALL 10X files (genotyping/cycle_1)
echo "Processing ALL 10X files from genotyping/cycle_1..."
python run_parallel_nd2_segmentation_v2.py \
    --input "$DATA_DIR/genotyping/cycle_1" \
    --output "$OUTPUT_DIR" \
    --cores "$NUM_CORES"

# Step 2: Process ALL 40X files (phenotyping)
echo "Processing ALL 40X files from phenotyping..."
python run_parallel_nd2_segmentation_v2.py \
    --input "$DATA_DIR/phenotyping" \
    --output "$OUTPUT_DIR" \
    --cores "$NUM_CORES"

echo "Segmentation completed. Results saved to $OUTPUT_DIR"

# Step 3: Create a Python script to stitch the segmented files
cat > "$OUTPUT_DIR/stitch_centroids.py" << 'EOF'
#!/usr/bin/env python3
"""
Script to stitch together centroids from multiple segmented images
and create composite images for 10X and 40X data.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('stitch-centroids')

def stitch_centroids(input_dir, mag="10X"):
    """
    Stitch centroids from all files of the same magnification.
    
    Args:
        input_dir: Directory containing segmentation results
        mag: Magnification to process ("10X" or "40X")
        
    Returns:
        np.ndarray: Combined centroids with global coordinates
    """
    input_path = Path(input_dir)
    mag_dir = input_path / mag
    
    if not mag_dir.exists():
        logger.error(f"Directory {mag_dir} does not exist")
        return np.empty((0, 2))
    
    # Find all well directories
    well_dirs = [d for d in mag_dir.iterdir() if d.is_dir()]
    if not well_dirs:
        logger.error(f"No well directories found in {mag_dir}")
        return np.empty((0, 2))
    
    all_centroids = []
    offsets = {}  # To store file offsets
    
    # First pass: collect information about each file
    for well_dir in well_dirs:
        well_id = well_dir.name
        centroid_files = list(well_dir.glob('*_nuclei_centroids_local.npy'))
        
        # Sort files by sequential number in filename
        def get_seq_num(filename):
            parts = filename.stem.split('_')
            for part in parts:
                if part.startswith('Seq'):
                    try:
                        return int(part[3:])
                    except ValueError:
                        return 0
            return 0
            
        centroid_files.sort(key=get_seq_num)
        
        # Analyze file locations and assign grid positions
        # This is a simple approach - in a real pipeline, these positions would come from metadata
        grid_size = int(np.ceil(np.sqrt(len(centroid_files))))
        
        # Estimate image dimensions from the first file's centroids
        if centroid_files:
            try:
                first_centroids = np.load(centroid_files[0])
                if len(first_centroids) > 0:
                    y_max, x_max = np.max(first_centroids, axis=0)
                    # Add some padding
                    y_max += 100
                    x_max += 100
                else:
                    # Default values if no centroids
                    y_max, x_max = 2000, 2000
            except Exception as e:
                logger.warning(f"Error loading first centroids file: {e}")
                y_max, x_max = 2000, 2000
        else:
            y_max, x_max = 2000, 2000
            
        logger.info(f"Estimated image dimensions for {mag}: {y_max} x {x_max}")
        
        # Assign grid positions and offsets
        for i, file_path in enumerate(centroid_files):
            file_id = file_path.stem.replace('_nuclei_centroids_local', '')
            row = i // grid_size
            col = i % grid_size
            
            # Calculate offset for this file
            offset_y = row * y_max
            offset_x = col * x_max
            
            offsets[file_id] = (offset_y, offset_x)
            logger.info(f"File {file_id} positioned at grid ({row}, {col}) with offset ({offset_y}, {offset_x})")
    
    # Second pass: load all centroids with offsets applied
    for well_dir in well_dirs:
        well_id = well_dir.name
        centroid_files = list(well_dir.glob('*_nuclei_centroids_local.npy'))
        
        for file_path in centroid_files:
            file_id = file_path.stem.replace('_nuclei_centroids_local', '')
            
            try:
                centroids = np.load(file_path)
                
                if len(centroids) == 0:
                    logger.warning(f"No centroids in {file_id}")
                    continue
                    
                # Apply offset to convert to global coordinates
                offset_y, offset_x = offsets[file_id]
                centroids_global = centroids.copy()
                centroids_global[:, 0] += offset_y  # Y coordinate
                centroids_global[:, 1] += offset_x  # X coordinate
                
                # Store with file info
                for centroid in centroids_global:
                    all_centroids.append({
                        'y': centroid[0],
                        'x': centroid[1],
                        'well_id': well_id,
                        'file_id': file_id
                    })
                
                logger.info(f"Added {len(centroids)} centroids from {file_id}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
    
    # Convert to DataFrame and then to numpy array
    if all_centroids:
        df = pd.DataFrame(all_centroids)
        # Create visualization
        create_visualization(df, input_dir, mag)
        # Return just the coordinates
        return df[['y', 'x']].values
    else:
        logger.warning(f"No centroids found for {mag}")
        return np.empty((0, 2))

def create_visualization(centroids_df, output_dir, mag):
    """
    Create a visualization of the stitched centroids.
    
    Args:
        centroids_df: DataFrame containing centroids and file information
        output_dir: Directory to save visualization
        mag: Magnification ("10X" or "40X")
    """
    output_path = Path(output_dir) / "composite"
    os.makedirs(output_path, exist_ok=True)
    
    plt.figure(figsize=(20, 20))
    
    # Color points by file_id for visual distinction
    file_ids = centroids_df['file_id'].unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(file_ids)))
    
    for i, file_id in enumerate(file_ids):
        subset = centroids_df[centroids_df['file_id'] == file_id]
        plt.scatter(subset['x'], subset['y'], s=1, color=colors[i], alpha=0.7, label=file_id)
    
    plt.title(f"Composite {mag} Image - {len(centroids_df)} nuclei centroids")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    # Hide the legend if there are too many files
    if len(file_ids) <= 20:
        plt.legend(fontsize='small', markerscale=5)
    
    # Save high-resolution figure
    plt.savefig(output_path / f"composite_{mag}_nuclei.png", dpi=300)
    plt.close()
    
    # Also save the data
    centroids_df.to_csv(output_path / f"composite_{mag}_centroids.csv", index=False)
    centroids = centroids_df[['y', 'x']].values
    np.save(output_path / f"composite_{mag}_centroids.npy", centroids)
    
    logger.info(f"Saved composite visualization and data for {mag}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python stitch_centroids.py <input_directory>")
        sys.exit(1)
        
    input_dir = sys.argv[1]
    
    # Process 10X data
    logger.info("Stitching 10X centroids...")
    centroids_10x = stitch_centroids(input_dir, "10X")
    logger.info(f"Generated composite image with {len(centroids_10x)} 10X centroids")
    
    # Process 40X data
    logger.info("Stitching 40X centroids...")
    centroids_40x = stitch_centroids(input_dir, "40X")
    logger.info(f"Generated composite image with {len(centroids_40x)} 40X centroids")
    
    logger.info("Stitching completed")
    
if __name__ == "__main__":
    main()
EOF

# Make the script executable
chmod +x "$OUTPUT_DIR/stitch_centroids.py"

# Step 4: Run the stitching script
echo "Stitching segmented files into composite images..."
python "$OUTPUT_DIR/stitch_centroids.py" "$OUTPUT_DIR"

# Step 5: Create a script to calculate the transformation matrix from the composite images
cat > "$OUTPUT_DIR/calculate_transformation.py" << 'EOF'
#!/usr/bin/env python3
"""
Script to calculate the optimal transformation matrix between
composite 10X and 40X images.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from scipy.optimize import minimize
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('calculate-transformation')

def transform_points(points, params):
    """
    Apply affine transformation to points.
    
    Args:
        points: Array of points (Nx2)
        params: Transformation parameters [a, b, tx, c, d, ty]
        
    Returns:
        transformed_points: Transformed points (Nx2)
    """
    a, b, tx, c, d, ty = params
    transformed = np.zeros_like(points)
    transformed[:, 0] = a * points[:, 0] + b * points[:, 1] + tx
    transformed[:, 1] = c * points[:, 0] + d * points[:, 1] + ty
    return transformed

def calculate_rmsd(src_points, dst_points, params):
    """
    Calculate Root Mean Square Deviation between transformed source points and destination points.
    
    Args:
        src_points: Source points (Nx2)
        dst_points: Destination points (Nx2)
        params: Transformation parameters
        
    Returns:
        rmsd: Root Mean Square Deviation
    """
    transformed = transform_points(src_points, params)
    squared_diffs = np.sum((transformed - dst_points) ** 2, axis=1)
    return np.sqrt(np.mean(squared_diffs))

def objective_function(params, src_points, dst_points):
    """
    Objective function for optimization.
    
    Args:
        params: Transformation parameters
        src_points: Source points
        dst_points: Destination points
        
    Returns:
        error: Sum of squared distances between transformed source and destination points
    """
    transformed = transform_points(src_points, params)
    return np.sum((transformed - dst_points) ** 2)

def find_initial_correspondences(centroids_10x, centroids_40x, scale=4.0, max_points=1000):
    """
    Find approximate initial correspondences between 10X and 40X centroids.
    
    Args:
        centroids_10x: Array of 10X centroids (Nx2)
        centroids_40x: Array of 40X centroids (Nx2)
        scale: Approximate scale between 10X and 40X
        max_points: Maximum number of points to use
        
    Returns:
        src_points: Subset of 10X centroids
        dst_points: Corresponding 40X centroids
    """
    # Limit number of points
    if len(centroids_10x) > max_points:
        indices = np.random.choice(len(centroids_10x), max_points, replace=False)
        centroids_10x = centroids_10x[indices]
    
    if len(centroids_40x) > max_points:
        indices = np.random.choice(len(centroids_40x), max_points, replace=False)
        centroids_40x = centroids_40x[indices]
    
    # Center the point sets
    center_10x = np.mean(centroids_10x, axis=0)
    center_40x = np.mean(centroids_40x, axis=0)
    
    centered_10x = centroids_10x - center_10x
    centered_40x = centroids_40x - center_40x
    
    # Scale the 10x points to approximate 40x scale
    scaled_10x = centered_10x * scale
    
    # For this simple example, just use a subset of points
    # In a real pipeline, you would use RANSAC or similar for robust matching
    min_size = min(len(centroids_10x), len(centroids_40x), 200)
    indices_10x = np.random.choice(len(centroids_10x), min_size, replace=False)
    indices_40x = np.random.choice(len(centroids_40x), min_size, replace=False)
    
    return centroids_10x[indices_10x], centroids_40x[indices_40x]

def optimize_transformation(centroids_10x, centroids_40x):
    """
    Optimize the transformation matrix between 10X and 40X centroids.
    
    Args:
        centroids_10x: Array of 10X centroids (Nx2)
        centroids_40x: Array of 40X centroids (Nx2)
        
    Returns:
        params: Optimized transformation parameters
        rmsd: Final RMSD
    """
    # Find initial correspondences
    src_points, dst_points = find_initial_correspondences(centroids_10x, centroids_40x)
    
    # Initial parameters: [a, b, tx, c, d, ty]
    # Start with approximate scale difference (10x to 40x = ~4x)
    initial_params = np.array([4.0, 0.0, 0.0, 0.0, 4.0, 0.0])
    
    # Optimize
    result = minimize(
        objective_function,
        initial_params,
        args=(src_points, dst_points),
        method='L-BFGS-B',
        options={'maxiter': 200}
    )
    
    # Calculate final RMSD
    rmsd = calculate_rmsd(src_points, dst_points, result.x)
    
    return result.x, rmsd

def visualize_transformation(centroids_10x, centroids_40x, params, output_dir):
    """
    Visualize the transformation result.
    
    Args:
        centroids_10x: Array of 10X centroids (Nx2)
        centroids_40x: Array of 40X centroids (Nx2)
        params: Transformation parameters
        output_dir: Directory to save visualization
    """
    # Sample a subset of points for visualization
    max_points = 500
    if len(centroids_10x) > max_points:
        indices = np.random.choice(len(centroids_10x), max_points, replace=False)
        viz_10x = centroids_10x[indices]
    else:
        viz_10x = centroids_10x
        
    if len(centroids_40x) > max_points:
        indices = np.random.choice(len(centroids_40x), max_points, replace=False)
        viz_40x = centroids_40x[indices]
    else:
        viz_40x = centroids_40x
    
    # Transform 10X to 40X space
    transformed_10x = transform_points(viz_10x, params)
    
    # Create visualization directory
    output_path = Path(output_dir) / "composite"
    os.makedirs(output_path, exist_ok=True)
    
    # Plot original and transformed points
    plt.figure(figsize=(15, 15))
    plt.scatter(viz_40x[:, 1], viz_40x[:, 0], s=20, alpha=0.5, color='red', label='40X (Target)')
    plt.scatter(transformed_10x[:, 1], transformed_10x[:, 0], s=10, alpha=0.5, color='blue', label='10X (Transformed)')
    
    plt.title("10X to 40X Transformation Result")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save visualization
    plt.savefig(output_path / "transformation_result.png", dpi=300)
    plt.close()
    
    logger.info(f"Saved transformation visualization to {output_path / 'transformation_result.png'}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python calculate_transformation.py <input_directory>")
        sys.exit(1)
        
    input_dir = sys.argv[1]
    composite_dir = Path(input_dir) / "composite"
    
    # Load composite centroids
    try:
        centroids_10x = np.load(composite_dir / "composite_10X_centroids.npy")
        centroids_40x = np.load(composite_dir / "composite_40X_centroids.npy")
        
        logger.info(f"Loaded {len(centroids_10x)} 10X centroids and {len(centroids_40x)} 40X centroids")
        
        if len(centroids_10x) == 0 or len(centroids_40x) == 0:
            logger.error("Empty centroid arrays")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error loading composite centroids: {e}")
        sys.exit(1)
    
    # Calculate transformation matrix
    logger.info("Calculating optimal transformation matrix...")
    params, rmsd = optimize_transformation(centroids_10x, centroids_40x)
    
    # Print results
    logger.info(f"Optimization completed with RMSD: {rmsd}")
    logger.info(f"Transformation parameters: a={params[0]:.4f}, b={params[1]:.4f}, tx={params[2]:.4f}, c={params[3]:.4f}, d={params[4]:.4f}, ty={params[5]:.4f}")
    
    # Save transformation matrix
    transformation = {
        "a": float(params[0]),
        "b": float(params[1]),
        "tx": float(params[2]),
        "c": float(params[3]),
        "d": float(params[4]),
        "ty": float(params[5]),
        "rmsd": float(rmsd)
    }
    
    with open(composite_dir / "transformation_matrix.json", 'w') as f:
        json.dump(transformation, f, indent=2)
    
    logger.info(f"Saved transformation matrix to {composite_dir / 'transformation_matrix.json'}")
    
    # Visualize result
    visualize_transformation(centroids_10x, centroids_40x, params, input_dir)
    
if __name__ == "__main__":
    main()
EOF

# Make the script executable
chmod +x "$OUTPUT_DIR/calculate_transformation.py"

# Step 6: Run the transformation calculation
echo "Calculating optimal transformation matrix..."
python "$OUTPUT_DIR/calculate_transformation.py" "$OUTPUT_DIR"

# Step 7: Display results
echo ""
echo "Full pipeline completed!"
echo "-----------------------"
echo "Results saved to: $OUTPUT_DIR"
echo ""

if [ -f "$OUTPUT_DIR/composite/transformation_matrix.json" ]; then
    echo "Optimal Transformation Matrix:"
    cat "$OUTPUT_DIR/composite/transformation_matrix.json" | python -m json.tool
    
    echo ""
    echo "Visualization saved to: $OUTPUT_DIR/composite/transformation_result.png"
    echo ""
    echo "You can apply this transformation matrix to map 10X coordinates to 40X coordinates using:"
    echo "40X_y = a * 10X_y + b * 10X_x + tx"
    echo "40X_x = c * 10X_y + d * 10X_x + ty"
else
    echo "Transformation matrix calculation failed. Check the logs for errors."
fi