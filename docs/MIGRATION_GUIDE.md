# Migration Guide: From Legacy Pipeline to ImageAnalysis Package

This guide helps users transition from the original pipeline scripts to the modern `imageanalysis` Python package.

## Overview

The ImageAnalysis project has evolved from a collection of standalone scripts to a professional Python package. This guide will help you:
- Understand the key differences
- Map old functionality to new
- Update your workflows
- Migrate existing code

## Quick Start

### Installation

**Old Way:**
```bash
# Copy scripts to your directory
cp -r /path/to/original_pipeline/* ./
# Manually install dependencies
conda install numpy pandas scikit-image cellpose nd2reader
```

**New Way:**
```bash
# Install package with all dependencies
pip install -e .
# Or for specific environments
conda env create -f environment.yml
conda activate ImageAnalysis
```

### Running Pipelines

**Old Way:**
```bash
# Run individual scripts
python Segment_10X.py /path/to/images /path/to/output
python Genotyping_Pipeline.py /path/to/segmented /path/to/output
python Phenotype_Cells.py /path/to/data /path/to/output
```

**New Way:**
```bash
# Use command-line tools
run_segmentation /path/to/images /path/to/output --magnification 10x
run_genotyping /path/to/segmented /path/to/output --config config.yml
run_phenotyping /path/to/data /path/to/output --config config.yml
```

## Detailed Migration

### 1. Segmentation

**Old Pipeline:**
```python
# Segment_10X.py or Segment_40X.py
python Segment_10X.py input_dir output_dir --diameter 30 --flow 0.4
```

**New Package:**
```python
# Command line
run_segmentation input_dir output_dir --magnification 10x --config config.yml

# Python API
from imageanalysis.core.segmentation import Segmentation10XPipeline

pipeline = Segmentation10XPipeline(
    input_file="image.nd2",
    output_dir="output/",
    cell_diameter=30,
    flow_threshold=0.4
)
result = pipeline.run()
```

### 2. Genotyping

**Old Pipeline:**
```python
# Genotyping_Pipeline.py
python Genotyping_Pipeline.py \
    --data_dir /path/to/cycles \
    --barcode_file barcodes.csv \
    --output_dir results/
```

**New Package:**
```python
# Command line
run_genotyping /path/to/cycles results/ --barcode-library barcodes.csv

# Python API
from imageanalysis.core.genotyping.pipeline import StandardGenotypingPipeline

pipeline = StandardGenotypingPipeline(
    data_dir="/path/to/cycles",
    barcode_library="barcodes.csv",
    output_dir="results/"
)
pipeline.run()
```

### 3. Phenotyping

**Old Pipeline:**
```python
# Phenotype_Cells.py
python Phenotype_Cells.py \
    --phenotype_dir /path/to/phenotype \
    --segmentation_dir /path/to/masks \
    --output_dir results/
```

**New Package:**
```python
# Command line
run_phenotyping /path/to/phenotype results/ --masks /path/to/masks

# Python API
from imageanalysis.core.phenotyping.pipeline import PhenotypingPipeline

pipeline = PhenotypingPipeline(
    phenotype_dir="/path/to/phenotype",
    segmentation_dir="/path/to/masks",
    output_dir="results/"
)
pipeline.run()
```

### 4. Image Registration/Mapping

**Old Pipeline:**
```python
# Run Jupyter notebooks manually
# Mapping_1_Cal_M_Matrix.ipynb
# Mapping_2_Find_Fiducials.ipynb
# Mapping_3_Optimize_Mapping_DOF.ipynb
```

**New Package:**
```python
# Command line
run_mapping /path/to/10x /path/to/40x output/ --config mapping_config.yml

# Python API
from imageanalysis.core.mapping.pipeline import MappingPipeline

pipeline = MappingPipeline(
    low_mag_dir="/path/to/10x",
    high_mag_dir="/path/to/40x",
    output_dir="output/"
)
mapping_result = pipeline.run()
```

## Function Mapping

| Old Function (In_Situ_Functions.py) | New Location |
|-------------------------------------|--------------|
| `load_nd2()` | `imageanalysis.utils.io.ImageLoader.load_image()` |
| `segment_nuclei()` | `imageanalysis.core.segmentation.base.segment_nuclei()` |
| `call_peaks()` | `imageanalysis.core.genotyping.peak_calling.call_peaks()` |
| `assign_barcodes()` | `imageanalysis.core.genotyping.barcode_assignment.assign_barcodes()` |
| `measure_phenotype()` | `imageanalysis.core.phenotyping.metrics.calculate_metrics()` |
| `create_album()` | `imageanalysis.core.visualization.albums.create_album()` |

## Configuration Files

**Old Way:** Command-line arguments for each parameter
```bash
python Segment_10X.py --diameter 30 --flow 0.4 --cellprob 0.0
```

**New Way:** YAML configuration files
```yaml
# config.yml
segmentation:
  cell_diameter: 30
  flow_threshold: 0.4
  cellprob_threshold: 0.0
  
genotyping:
  cycles: 10
  channels: ['G-ISS', 'T-ISS', 'A-ISS', 'C-ISS']
  
phenotyping:
  channels: ['DAPI', 'mClov3', 'TMR']
  metrics: ['mean_intensity', 'total_intensity']
```

## Path Handling

**Old Way:** Hardcoded or relative paths
```python
data_dir = "../data/phenotyping/"
output_dir = "./results/"
```

**New Way:** Pathlib and absolute paths
```python
from pathlib import Path

data_dir = Path("/absolute/path/to/data")
output_dir = Path("/absolute/path/to/output")
```

## Error Handling

**Old Way:** Basic try/except or no error handling
```python
try:
    process_image(image)
except:
    print("Error processing image")
```

**New Way:** Comprehensive error handling with logging
```python
import logging
from imageanalysis.utils.logging import setup_logger

logger = setup_logger(__name__)

try:
    result = pipeline.run()
except FileNotFoundError as e:
    logger.error(f"Input file not found: {e}")
except ValueError as e:
    logger.error(f"Invalid parameter: {e}")
```

## Parallel Processing

**Old Way:** Sequential processing or manual parallelization
```bash
# Submit multiple SLURM jobs
sbatch Segment_10X.sh image1.nd2
sbatch Segment_10X.sh image2.nd2
```

**New Way:** Built-in parallel processing
```python
from imageanalysis.core.pipeline import BatchPipeline

batch = BatchPipeline(
    pipeline_class=Segmentation10XPipeline,
    n_workers=4
)
batch.process_directory(input_dir, output_dir)
```

## Output Formats

**Old Way:** Various formats (CSV, NPY, custom)
**New Way:** Standardized outputs with metadata

```python
# All pipelines now include:
# - Standardized file naming
# - JSON metadata files
# - Consistent CSV formats
# - Optional visualization outputs
```

## Testing Your Migration

1. **Start with a small dataset** to verify results match
2. **Compare outputs** between old and new pipelines
3. **Check performance** - new pipeline should be faster
4. **Validate results** using the test suite:
   ```bash
   pytest tests/integration/test_pipeline_comparison.py
   ```

## Common Issues

### Import Errors
```python
# Old
from In_Situ_Functions import segment_nuclei

# New  
from imageanalysis.core.segmentation import segment_nuclei
```

### File Path Issues
- Use absolute paths or Path objects
- Check that file extensions are correct (.nd2, .tif, etc.)

### Missing Dependencies
```bash
# Ensure all dependencies are installed
pip install -e ".[all]"
```

### Different Results
- Check configuration parameters match
- Verify image preprocessing is consistent
- Some algorithms may have been improved

## Getting Help

1. Check the [package documentation](../README.md)
2. Review the [test examples](../tests/)
3. Look at the [legacy code](../legacy/) for reference
4. Submit issues on GitHub for bugs or questions

## Benefits of Migration

✅ **Easier installation** - Single pip command  
✅ **Better performance** - Optimized algorithms  
✅ **Comprehensive testing** - Reliable results  
✅ **Modern Python** - Type hints, pathlib, logging  
✅ **Extensible** - Easy to add new features  
✅ **Maintainable** - Clean, documented code  
✅ **Reproducible** - Configuration files and version control