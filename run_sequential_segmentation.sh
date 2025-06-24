#!/bin/bash
# Sequential ND2 segmentation and global coordinate mapping

# Get the number of available cores for parallel processing
NUM_CORES=$(($(sysctl -n hw.ncpu) - 1))
echo "Running with $NUM_CORES cores"

# Set base directories
DATA_DIR="/Users/csm70/Desktop/Shalem_Lab/ImageAnalysis/Data"
OUTPUT_DIR="/Users/csm70/Desktop/Shalem_Lab/ImageAnalysis/results/nd2_sequential"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/10X"
mkdir -p "$OUTPUT_DIR/40X"
mkdir -p "$OUTPUT_DIR/global_coords"

# Create a Python script to handle sequential processing with global coordinates
cat > "$OUTPUT_DIR/sequential_segmentation.py" << 'EOF'
#!/usr/bin/env python3
"""
Sequential ND2 segmentation with global coordinate mapping.
Processes each image individually, then maps its nuclei coordinates to a global space.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import json
import logging
import time
from pathlib import Path
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import tracemalloc

# Import segmentation functionality 
from nd2_segmentation_v2 import run_nd2_segmentation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sequential-nd2-segmentation')

def find_nd2_files(input_dir, pattern="*.nd2"):
    """Find all ND2 files in the input directory.
    
    Args:
        input_dir: Directory to search for ND2 files
        pattern: Pattern to match ND2 files
        
    Returns:
        list: List of ND2 file paths
    """
    input_path = Path(input_dir)
    return list(input_path.glob(f"**/{pattern}"))

def get_magnification_from_path(file_path):
    """Determine the magnification from the file path.
    
    Args:
        file_path: Path to the ND2 file
        
    Returns:
        str: '10X' or '40X'
    """
    path_str = str(file_path).lower()
    
    # Try to determine from directory structure
    if "genotyping" in path_str or "10x" in path_str:
        return "10X"
    elif "phenotyping" in path_str or "40x" in path_str:
        return "40X"
    
    # Default to 10X
    return "10X"

def calculate_grid_position(file_path, file_index, grid_size=None):
    """
    Calculate the grid position for a file based on its filename or index.
    
    Args:
        file_path: Path to the file
        file_index: Index of the file in the list
        grid_size: Optional grid size (will be calculated if not provided)
        
    Returns:
        tuple: (row, col) position in the grid
    """
    try:
        # Try to extract sequence number from filename
        file_name = file_path.name
        parts = file_name.split('_')
        seq_num = None
        
        for part in parts:
            if part.startswith('Seq'):
                try:
                    seq_num = int(part[3:])
                    break
                except ValueError:
                    pass
        
        # If we got a valid sequence number, use it to determine position
        if seq_num is not None:
            if grid_size is None:
                # Default to a reasonable grid size
                grid_size = 10
                
            # Calculate grid position
            row = (seq_num // 100) % grid_size  # Group by hundreds
            col = (seq_num % 100) // 10         # Group by tens
            return row, col
        
        # Fall back to using file_index
        if grid_size is None:
            grid_size = int(np.ceil(np.sqrt(file_index + 1)))
            
        row = file_index // grid_size
        col = file_index % grid_size
        return row, col
        
    except Exception as e:
        logger.warning(f"Error calculating grid position: {e}")
        # Default fallback - just use file_index
        if grid_size is None:
            grid_size = int(np.ceil(np.sqrt(file_index + 1)))
            
        row = file_index // grid_size
        col = file_index % grid_size
        return row, col

def process_file(nd2_file, output_dir, file_index, magnification=None, grid_size=None):
    """
    Process a single ND2 file and map its coordinates to global space.
    
    Args:
        nd2_file: Path to ND2 file
        output_dir: Directory to save results
        file_index: Index of the file in the processing list
        magnification: Optional override for magnification
        grid_size: Optional grid size for positioning
        
    Returns:
        dict: Results of segmentation including global coordinates
    """
    try:
        start_time = time.time()
        file_path = Path(nd2_file)
        file_id = file_path.stem
        
        # Determine magnification if not provided
        if magnification is None:
            magnification = get_magnification_from_path(file_path)
        
        # Create full output directory for local results
        local_output_dir = os.path.join(output_dir, magnification)
        
        logger.info(f"Processing {file_id} at {magnification} magnification (index {file_index})")
        
        # Run segmentation
        results = run_nd2_segmentation(
            nd2_file=str(file_path),
            output_dir=local_output_dir,
            magnification=magnification,
            nuclear_channel=0,  # DAPI usually channel 0
            cell_channel=1      # Cell marker usually channel 1
        )
        
        # If segmentation failed, return error
        if results.get("status") != "success":
            logger.error(f"Segmentation failed for {file_id}: {results.get('error', 'Unknown error')}")
            return {
                "status": "error",
                "file": str(file_path),
                "magnification": magnification,
                "error": results.get("error", "Unknown error"),
                "processing_time": time.time() - start_time
            }
        
        # Get well ID from results
        well_id = results.get("well", "Well1")
        
        # Calculate grid position
        row, col = calculate_grid_position(file_path, file_index, grid_size)
        
        # Define image dimensions based on magnification
        # These are approximate and might need tuning for your specific images
        if magnification == "10X":
            image_height, image_width = 1024, 1024  # Example dimensions for 10X
        else:  # 40X
            image_height, image_width = 2304, 2304  # Example dimensions for 40X
            
        # Add some padding between images in the grid
        padding = 100
        
        # Calculate global offset for this image
        offset_y = row * (image_height + padding)
        offset_x = col * (image_width + padding)
        
        # Load local centroids to convert to global coordinates
        centroids_file = os.path.join(local_output_dir, well_id, f"{file_id}_nuclei_centroids_local.npy")
        
        try:
            local_centroids = np.load(centroids_file)
            num_centroids = len(local_centroids)
            
            if num_centroids > 0:
                # Create global centroids by adding offsets
                global_centroids = local_centroids.copy()
                global_centroids[:, 0] += offset_y  # Y coordinate
                global_centroids[:, 1] += offset_x  # X coordinate
                
                # Create dataframe with all information
                centroids_df = pd.DataFrame({
                    'local_y': local_centroids[:, 0],
                    'local_x': local_centroids[:, 1],
                    'global_y': global_centroids[:, 0],
                    'global_x': global_centroids[:, 1],
                    'file_id': file_id,
                    'well_id': well_id,
                    'magnification': magnification,
                    'grid_row': row,
                    'grid_col': col
                })
                
                # Save global centroids
                global_dir = os.path.join(output_dir, "global_coords", magnification)
                os.makedirs(global_dir, exist_ok=True)
                
                # Save as CSV for easier inspection
                csv_file = os.path.join(global_dir, f"{file_id}_global_centroids.csv")
                centroids_df.to_csv(csv_file, index=False)
                
                # Save just the global coordinates as numpy
                npy_file = os.path.join(global_dir, f"{file_id}_global_centroids.npy")
                np.save(npy_file, global_centroids)
                
                logger.info(f"Mapped {num_centroids} centroids from {file_id} to global coordinates at grid position ({row}, {col})")
                
                # Add global coordinate info to results
                results.update({
                    "global_mapping": {
                        "grid_row": int(row),
                        "grid_col": int(col),
                        "offset_y": int(offset_y),
                        "offset_x": int(offset_x),
                        "global_centroids_file": npy_file,
                        "num_global_centroids": num_centroids
                    }
                })
                
                # Save a small visualization of this image's centroids in global space
                plt.figure(figsize=(6, 6))
                plt.scatter(global_centroids[:, 1], global_centroids[:, 0], s=5, alpha=0.7)
                plt.title(f"{file_id} - Grid Position ({row}, {col})")
                plt.xlabel("Global X")
                plt.ylabel("Global Y")
                
                viz_dir = os.path.join(global_dir, "visualizations")
                os.makedirs(viz_dir, exist_ok=True)
                plt.savefig(os.path.join(viz_dir, f"{file_id}_global.png"), dpi=100)
                plt.close()
            else:
                logger.warning(f"No centroids found in {file_id}")
                results.update({
                    "global_mapping": {
                        "grid_row": int(row),
                        "grid_col": int(col),
                        "offset_y": int(offset_y),
                        "offset_x": int(offset_x),
                        "num_global_centroids": 0
                    }
                })
                
        except Exception as e:
            logger.error(f"Error processing centroids for {file_id}: {e}")
            results.update({
                "global_mapping_error": str(e)
            })
        
        # Save expanded results with global mapping info
        results_file = os.path.join(local_output_dir, well_id, f"{file_id}_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Add processing time
        results["processing_time"] = time.time() - start_time
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing {nd2_file}: {e}")
        return {
            "status": "error",
            "file": str(file_path),
            "error": str(e),
            "processing_time": time.time() - start_time
        }

def generate_composite_visualization(output_dir, magnification):
    """
    Generate a lightweight visualization of all centroids in global space.
    
    Args:
        output_dir: Base output directory
        magnification: Magnification to visualize
    """
    global_dir = os.path.join(output_dir, "global_coords", magnification)
    centroid_files = list(Path(global_dir).glob("*_global_centroids.npy"))
    
    if not centroid_files:
        logger.warning(f"No global centroid files found for {magnification}")
        return
    
    # Collect all global centroids
    all_centroids = []
    all_file_ids = []
    
    for file_path in centroid_files:
        try:
            centroids = np.load(file_path)
            if len(centroids) > 0:
                all_centroids.append(centroids)
                file_id = file_path.stem.replace('_global_centroids', '')
                all_file_ids.append(file_id)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    if not all_centroids:
        logger.warning(f"No valid centroids loaded for {magnification}")
        return
    
    # Create visualization with different colors for each file
    plt.figure(figsize=(20, 20))
    
    # Use a colormap
    colors = plt.cm.rainbow(np.linspace(0, 1, len(all_centroids)))
    
    # Plot each file's centroids with a different color
    for i, (centroids, file_id) in enumerate(zip(all_centroids, all_file_ids)):
        # Downsample if there are too many points to keep the plot responsive
        if len(centroids) > 1000:
            indices = np.random.choice(len(centroids), 1000, replace=False)
            plot_centroids = centroids[indices]
        else:
            plot_centroids = centroids
            
        plt.scatter(
            plot_centroids[:, 1], 
            plot_centroids[:, 0], 
            s=2, 
            color=colors[i], 
            alpha=0.5, 
            label=file_id if i < 20 else None  # Only show legend for first 20 files
        )
    
    # Calculate total number of centroids
    total_centroids = sum(len(c) for c in all_centroids)
    
    plt.title(f"Composite {magnification} Image - {total_centroids} nuclei centroids from {len(all_centroids)} files")
    plt.xlabel("Global X coordinate")
    plt.ylabel("Global Y coordinate")
    
    # Only show legend if there aren't too many files
    if len(all_centroids) <= 20:
        plt.legend(fontsize='small', markerscale=2, loc='upper right')
    
    # Save the visualization
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    viz_path = os.path.join(output_dir, "visualizations", f"composite_{magnification}_nuclei.png")
    plt.savefig(viz_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved composite visualization for {magnification} to {viz_path}")
    
    # Also create a summary DataFrame with all centroid information
    summary_data = []
    
    for file_path in Path(global_dir).glob("*_global_centroids.csv"):
        try:
            df = pd.read_csv(file_path)
            summary_data.append(df)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
    
    if summary_data:
        all_data = pd.concat(summary_data, ignore_index=True)
        summary_path = os.path.join(output_dir, "global_coords", f"all_{magnification}_centroids.csv")
        all_data.to_csv(summary_path, index=False)
        
        # Also save just the global coordinates as numpy for optimization
        global_coords = all_data[['global_y', 'global_x']].values
        np.save(os.path.join(output_dir, "global_coords", f"all_{magnification}_centroids.npy"), global_coords)
        
        logger.info(f"Saved combined data for {magnification} with {len(all_data)} centroids")
        return global_coords
    else:
        logger.warning(f"No centroid CSV files found for {magnification}")
        return None

def optimize_transformation(points_10x, points_40x):
    """
    Calculate the optimal transformation from 10X to 40X coordinates.
    
    Args:
        points_10x: 10X points (Nx2)
        points_40x: 40X points (Nx2)
        
    Returns:
        dict: Transformation parameters and RMSD
    """
    from scipy.optimize import minimize
    
    # Transform points function
    def transform_points(points, params):
        a, b, tx, c, d, ty = params
        transformed = np.zeros_like(points)
        transformed[:, 0] = a * points[:, 0] + b * points[:, 1] + tx
        transformed[:, 1] = c * points[:, 0] + d * points[:, 1] + ty
        return transformed
    
    # Objective function for optimization
    def objective(params, src, dst):
        transformed = transform_points(src, params)
        return np.sum((transformed - dst) ** 2)
    
    # Calculate RMSD
    def calculate_rmsd(src, dst, params):
        transformed = transform_points(src, params)
        squared_diffs = np.sum((transformed - dst) ** 2, axis=1)
        return np.sqrt(np.mean(squared_diffs))
    
    # Sample points (to keep optimization reasonable)
    max_points = 1000
    if len(points_10x) > max_points:
        indices = np.random.choice(len(points_10x), max_points, replace=False)
        sample_10x = points_10x[indices]
    else:
        sample_10x = points_10x
        
    if len(points_40x) > max_points:
        indices = np.random.choice(len(points_40x), max_points, replace=False)
        sample_40x = points_40x[indices]
    else:
        sample_40x = points_40x
    
    # Initial guess: Scale is approximately 4x from 10X to 40X
    initial_params = np.array([4.0, 0.0, 0.0, 0.0, 4.0, 0.0])
    
    # Run the optimization
    logger.info(f"Optimizing transformation with {len(sample_10x)} 10X points and {len(sample_40x)} 40X points")
    
    tracemalloc.start()
    start_time = time.time()
    
    result = minimize(
        objective,
        initial_params,
        args=(sample_10x, sample_40x),
        method='L-BFGS-B',
        options={'maxiter': 200}
    )
    
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Calculate final RMSD
    rmsd = calculate_rmsd(sample_10x, sample_40x, result.x)
    
    logger.info(f"Optimization completed: RMSD={rmsd:.4f}, Time={(end_time-start_time)*1000:.2f}ms, Memory={peak/1024/1024:.2f}MB")
    
    # Return parameters and RMSD
    return {
        "a": float(result.x[0]),
        "b": float(result.x[1]),
        "tx": float(result.x[2]),
        "c": float(result.x[3]),
        "d": float(result.x[4]),
        "ty": float(result.x[5]),
        "rmsd": float(rmsd),
        "iterations": int(result.nit),
        "success": bool(result.success),
        "time_ms": float((end_time - start_time) * 1000),
        "memory_mb": float(peak / 1024 / 1024)
    }

def main():
    parser = argparse.ArgumentParser(description='Sequential ND2 segmentation with global coordinate mapping')
    parser.add_argument('--input-10x', required=True, help='Directory containing 10X ND2 files')
    parser.add_argument('--input-40x', required=True, help='Directory containing 40X ND2 files')
    parser.add_argument('--output', required=True, help='Directory to save segmentation results')
    parser.add_argument('--cores', type=int, default=None, help='Number of CPU cores to use. Default is all available cores - 1')
    parser.add_argument('--limit-10x', type=int, default=None, help='Maximum number of 10X files to process')
    parser.add_argument('--limit-40x', type=int, default=None, help='Maximum number of 40X files to process')
    
    args = parser.parse_args()
    
    # Determine number of cores to use
    if args.cores is None:
        import multiprocessing
        args.cores = max(1, multiprocessing.cpu_count() - 1)
    
    start_time = time.time()
    
    # Process 10X files
    logger.info(f"Finding 10X ND2 files in {args.input_10x}")
    nd2_files_10x = find_nd2_files(args.input_10x)
    
    if args.limit_10x and args.limit_10x > 0:
        nd2_files_10x = nd2_files_10x[:args.limit_10x]
    
    num_files_10x = len(nd2_files_10x)
    logger.info(f"Found {num_files_10x} 10X ND2 files")
    
    if num_files_10x == 0:
        logger.error(f"No 10X ND2 files found in {args.input_10x}")
    else:
        # Process files sequentially but with parallel processing
        results_10x = []
        
        # Determine grid size for 10X files
        grid_size_10x = int(np.ceil(np.sqrt(num_files_10x)))
        logger.info(f"Using grid size {grid_size_10x} for 10X files")
        
        # Create process function with fixed args
        process_fn_10x = partial(
            process_file, 
            output_dir=args.output,
            magnification="10X",
            grid_size=grid_size_10x
        )
        
        with ProcessPoolExecutor(max_workers=args.cores) as executor:
            futures = {executor.submit(process_fn_10x, nd2_file, i): i for i, nd2_file in enumerate(nd2_files_10x)}
            
            # Show progress
            for i, future in enumerate(as_completed(futures)):
                try:
                    result = future.result()
                    results_10x.append(result)
                    logger.info(f"Processed 10X file {i+1}/{num_files_10x}")
                except Exception as e:
                    logger.error(f"Error in 10X file processing: {e}")
        
        # Generate composite visualization for 10X
        logger.info("Generating composite visualization for 10X data")
        centroids_10x = generate_composite_visualization(args.output, "10X")
    
    # Process 40X files
    logger.info(f"Finding 40X ND2 files in {args.input_40x}")
    nd2_files_40x = find_nd2_files(args.input_40x)
    
    if args.limit_40x and args.limit_40x > 0:
        nd2_files_40x = nd2_files_40x[:args.limit_40x]
    
    num_files_40x = len(nd2_files_40x)
    logger.info(f"Found {num_files_40x} 40X ND2 files")
    
    if num_files_40x == 0:
        logger.error(f"No 40X ND2 files found in {args.input_40x}")
    else:
        # Process files sequentially but with parallel processing
        results_40x = []
        
        # Determine grid size for 40X files
        grid_size_40x = int(np.ceil(np.sqrt(num_files_40x)))
        logger.info(f"Using grid size {grid_size_40x} for 40X files")
        
        # Create process function with fixed args
        process_fn_40x = partial(
            process_file, 
            output_dir=args.output,
            magnification="40X",
            grid_size=grid_size_40x
        )
        
        with ProcessPoolExecutor(max_workers=args.cores) as executor:
            futures = {executor.submit(process_fn_40x, nd2_file, i): i for i, nd2_file in enumerate(nd2_files_40x)}
            
            # Show progress
            for i, future in enumerate(as_completed(futures)):
                try:
                    result = future.result()
                    results_40x.append(result)
                    logger.info(f"Processed 40X file {i+1}/{num_files_40x}")
                except Exception as e:
                    logger.error(f"Error in 40X file processing: {e}")
        
        # Generate composite visualization for 40X
        logger.info("Generating composite visualization for 40X data")
        centroids_40x = generate_composite_visualization(args.output, "40X")
    
    # Calculate transformation if we have both 10X and 40X data
    if num_files_10x > 0 and num_files_40x > 0 and centroids_10x is not None and centroids_40x is not None:
        logger.info("Calculating transformation matrix between 10X and 40X data")
        
        transformation = optimize_transformation(centroids_10x, centroids_40x)
        
        # Save the transformation
        os.makedirs(os.path.join(args.output, "transformation"), exist_ok=True)
        with open(os.path.join(args.output, "transformation", "transformation_matrix.json"), 'w') as f:
            json.dump(transformation, f, indent=2)
            
        logger.info(f"Saved transformation matrix with RMSD={transformation['rmsd']:.4f}")
        
        # Create a visualization of the transformation
        logger.info("Creating transformation visualization")
        
        # Sample points to keep the visualization manageable
        max_points = 1000
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
        
        # Transform 10X points to 40X space
        a, b, tx, c, d, ty = [
            transformation['a'], 
            transformation['b'], 
            transformation['tx'], 
            transformation['c'], 
            transformation['d'], 
            transformation['ty']
        ]
        
        transformed_10x = np.zeros_like(viz_10x)
        transformed_10x[:, 0] = a * viz_10x[:, 0] + b * viz_10x[:, 1] + tx
        transformed_10x[:, 1] = c * viz_10x[:, 0] + d * viz_10x[:, 1] + ty
        
        # Create the visualization
        plt.figure(figsize=(15, 15))
        plt.scatter(viz_40x[:, 1], viz_40x[:, 0], s=5, alpha=0.5, color='red', label='40X (Target)')
        plt.scatter(transformed_10x[:, 1], transformed_10x[:, 0], s=5, alpha=0.5, color='blue', label='10X→40X (Transformed)')
        
        plt.title(f"10X to 40X Transformation - RMSD={transformation['rmsd']:.4f}")
        plt.xlabel("Global X coordinate")
        plt.ylabel("Global Y coordinate")
        plt.legend()
        
        # Save the visualization
        viz_path = os.path.join(args.output, "transformation", "transformation_visualization.png")
        plt.savefig(viz_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved transformation visualization to {viz_path}")
    
    # Calculate total execution time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Sequential segmentation completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Save summary
    summary = {
        "10x_files_processed": num_files_10x,
        "40x_files_processed": num_files_40x,
        "10x_centroids": None if centroids_10x is None else len(centroids_10x),
        "40x_centroids": None if centroids_40x is None else len(centroids_40x),
        "transformation": None if 'transformation' not in locals() else transformation,
        "execution_time": f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    }
    
    with open(os.path.join(args.output, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
        
    logger.info("Summary saved. Pipeline complete!")
    
if __name__ == "__main__":
    main()
EOF

# Make the script executable
chmod +x "$OUTPUT_DIR/sequential_segmentation.py"

# Run the sequential processing script
echo "Starting sequential processing of 10X and 40X ND2 files..."

python "$OUTPUT_DIR/sequential_segmentation.py" \
    --input-10x "$DATA_DIR/genotyping/cycle_1" \
    --input-40x "$DATA_DIR/phenotyping" \
    --output "$OUTPUT_DIR" \
    --cores "$NUM_CORES" \
    --limit-10x 4 \  # Change these limits when ready for a full run
    --limit-40x 4

# Display results
echo ""
echo "Sequential segmentation pipeline completed!"
echo "-----------------------------------------"
echo "Results saved to: $OUTPUT_DIR"
echo ""

if [ -f "$OUTPUT_DIR/transformation/transformation_matrix.json" ]; then
    echo "Optimal Transformation Matrix:"
    cat "$OUTPUT_DIR/transformation/transformation_matrix.json" | python -m json.tool
    
    echo ""
    echo "Visualization saved to: $OUTPUT_DIR/transformation/transformation_visualization.png"
    echo ""
    echo "You can apply this transformation matrix to map 10X coordinates to 40X coordinates using:"
    echo "40X_y = a * 10X_y + b * 10X_x + tx"
    echo "40X_x = c * 10X_y + d * 10X_x + ty"
else
    echo "Transformation matrix calculation may have failed. Check logs for details."
fi