# Segmentation Parallelization

This document explains the implementation and benchmarking of parallel segmentation in the ImageAnalysis pipeline.

## Overview

Segmentation is one of the most compute-intensive parts of the image analysis pipeline. By parallelizing the segmentation process across multiple CPU cores, we can significantly reduce processing time when dealing with multiple images.

## Implementation

The parallelization of the segmentation process is achieved using Python's `concurrent.futures.ProcessPoolExecutor` to distribute the workload across multiple cores:

```python
def perform_parallel_segmentation(image_files, output_dir, is_10x=True, num_workers=None):
    """
    Perform segmentation on multiple images in parallel.
    
    Args:
        image_files: List of image files to process
        output_dir: Output directory for segmentation results
        is_10x: Whether the images are 10x (True) or 40x (False)
        num_workers: Number of parallel workers to use (default: CPU count - 1)
        
    Returns:
        results: List of segmentation results
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        
    # Create partial functions for segmentation
    if is_10x:
        seg_func = partial(perform_segmentation, output_dir=output_dir, is_10x=True)
    else:
        seg_func = partial(perform_segmentation, output_dir=output_dir, is_10x=False)
    
    # Process images in parallel
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks in parallel to maximize core usage
        futures = [executor.submit(seg_func, img_file) for img_file in image_files]
        
        # Process results as they complete
        for i, future in enumerate(futures):
            result = future.result()
            results.append(result)
            logger.info(f"Completed segmentation {i+1}/{len(futures)}")
    
    return results
```

## Benchmarking Tool

A dedicated benchmarking tool (`parallel_segmentation_benchmark.py`) has been created to:

1. Measure segmentation performance across different numbers of CPU cores
2. Determine the optimal number of worker processes for your specific hardware
3. Quantify the speedup gained through parallelization
4. Generate detailed reports and visualizations of the parallelization efficiency

## Key Benefits

1. **Reduced Processing Time**: Processes multiple images simultaneously rather than sequentially
2. **Optimal Resource Utilization**: Distributes workload evenly across available CPU cores
3. **Responsive System**: By default leaves one core free to maintain system responsiveness
4. **Scalability**: Automatically adapts to systems with different numbers of CPU cores

## Usage in Real Pipeline

In the real image analysis pipeline, parallelization is implemented for both 10x and 40x image segmentation:

```python
# Determine number of cores to use (leave one core free)
num_cores = max(1, multiprocessing.cpu_count() - 1)
logger.info(f"Using {num_cores} CPU cores for parallel segmentation")

# Submit both 10x and 40x segmentation tasks concurrently
with ProcessPoolExecutor(max_workers=num_cores) as executor:
    # Submit all tasks to maximize core utilization
    futures_10x = [executor.submit(seg_10x_func, img) for img in image_10x_files]
    futures_40x = [executor.submit(seg_40x_func, img) for img in image_40x_files]
    
    # Process all results (both 10x and 40x) as they complete
    all_futures = futures_10x + futures_40x
    for future in all_futures:
        result = future.result()
        # Process result...
```

## Performance Expectations

The benchmark will determine the optimal configuration for your specific hardware, but general expectations are:

- **Linear scaling** up to a certain number of cores, typically showing diminishing returns with very high core counts
- **Near-optimal efficiency** when using (CPU count - 1) workers, balancing performance and system responsiveness
- **Impact of image size**: Larger images generally benefit more from parallelization due to higher computation/overhead ratio

## Recommendations Based on Benchmark

Run the segmentation benchmark to receive customized recommendations for your specific hardware:

```bash
./run_segmentation_benchmark.sh
```

The benchmark will generate a detailed recommendation in `results/segmentation_benchmark/recommendation.txt` based on actual measurements from your system.

## Environment Optimization

For optimal parallel segmentation performance, the following environment variables are set:

```bash
# Set numerical libraries to use all available cores
export OPENBLAS_NUM_THREADS=$(sysctl -n hw.ncpu)
export MKL_NUM_THREADS=$(sysctl -n hw.ncpu)
export OMP_NUM_THREADS=$(sysctl -n hw.ncpu)

# Enable Python to use multiprocessing effectively
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
```

## Integration with Optimization Benchmarks

Once the segmentation phase is optimized with parallelization, the output can be fed into the optimization algorithm benchmarks to evaluate L-BFGS-B and LMA performance on the resulting nuclear centroids.