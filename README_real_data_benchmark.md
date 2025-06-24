# Real Data Optimizer Benchmark

This script benchmarks the L-BFGS-B and Levenberg-Marquardt (LMA) optimization algorithms using real nuclear centroid data extracted from your microscopy images.

## Overview

The benchmark:

1. Processes your 10x and 40x microscopy images through the segmentation pipeline
2. Extracts actual nuclear centroids from the segmented images
3. Finds corresponding points between magnifications
4. Evaluates both optimization algorithms on these real point sets
5. Compares performance metrics (RMSD, time, memory usage)

## Key Advantages of Real Data Testing

This approach offers several advantages over simulated data testing:

- **Realistic Point Distributions**: Uses the actual spatial distribution of nuclei in your samples
- **True Magnification Differences**: Captures the real-world transformation parameters between 10x and 40x
- **Authentic Noise Characteristics**: Includes actual microscopy noise and segmentation variation
- **Direct Application Relevance**: Results directly applicable to your mapping pipeline

## Usage

```bash
# Run with default settings
python benchmark_optimizers_real_data.py

# Specify custom data and output directories
python benchmark_optimizers_real_data.py --data-dir /path/to/data --output-dir results/custom_benchmark

# Run with more trials
python benchmark_optimizers_real_data.py --trials 5
```

## Output

The script generates:

1. **CSV Data**:
   - `optimizer_benchmark_real_data.csv`: Raw benchmark data for each trial
   - `optimizer_benchmark_real_data_summary.csv`: Statistical summary of the results

2. **Plots**:
   - `rmsd_comparison_real_data.png`: Comparison of RMSD values across trials
   - `time_comparison_real_data.png`: Comparison of computation times

## Implementation Notes

- **Point Correspondence**: Uses a simplified RANSAC approach to find corresponding points between magnifications
- **Initial Parameters**: Starts with scientifically reasonable parameters (scale factor ~4x for 10x→40x)
- **Performance Metrics**: RMSD measures the alignment quality between transformed 10x and 40x nuclear coordinates
- **Multiple Trials**: Runs multiple optimization attempts to account for variability

## Analysis Tips

When interpreting the results, consider:

1. **RMSD Values**: Lower values indicate better alignment of nuclear coordinates across magnifications
2. **Transformation Parameters**: Check if the optimized scale factors (a, d) are close to the expected 4x ratio
3. **Consistency**: Look for consistency in the transformation parameters across trials
4. **Time vs. Accuracy Trade-off**: Consider whether any speed difference justifies potential accuracy differences