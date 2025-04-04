# ImageAnalysis

A Python package for image analysis pipelines in the Shalem Lab.

## Overview

This package provides tools for image processing, cellular segmentation, genotyping, and phenotyping of microscopy data. It is organized into modules that can be run either independently or as part of a complete pipeline.

## Project Structure

This repository contains:

1. **imageanalysis/** - Main Python package with a clean, modular structure
2. **original_pipeline/** - Original scripts preserved for reference
3. **tests/** - Test suite for validating functionality

## Installation

```bash
# Clone the repository
git clone https://github.com/caseysm/ImageAnalysis.git
cd ImageAnalysis

# Setup conda environment
conda env create -f environment.yml
conda activate ImageAnalysis

# Install in development mode
pip install -e .
```

You can also use the provided installation script:

```bash
./install.sh
```

## Package Structure

```
imageanalysis/          # Main package directory
├── __init__.py         # Package initialization
├── bin/                # Command-line scripts
│   ├── run_segmentation.py
│   ├── run_genotyping.py
│   ├── run_phenotyping.py
│   └── create_albums.py
├── core/               # Core functionality
│   ├── pipeline.py     # Base pipeline class
│   ├── segmentation/   # Cell segmentation
│   ├── genotyping/     # Cell genotyping
│   ├── phenotyping/    # Cell phenotyping
│   ├── mapping/        # Image registration
│   └── visualization/  # Visualization tools
├── utils/              # Utility functions
│   ├── io.py           # I/O utilities
│   └── logging.py      # Logging utilities
└── config/             # Configuration
    └── settings.py     # Default settings
```

## Command-line Tools

After installation, you can use the following command-line tools:

- `run-segmentation`: Segment cells in microscopy images (10X and 40X)
- `run-mapping`: Find optimal coordinate transformation between 10X and 40X images
- `run-genotyping`: Assign genotypes to segmented cells
- `run-phenotyping`: Measure phenotypic features of cells
- `create-albums`: Create visual albums of cells

## Example Usage

### Using command-line tools

```bash
# 10X Segmentation
run-segmentation data/genotyping/cycle_1/Well1_Point1_0034.nd2 --magnification 10x --output-dir results/segmentation_10x/Well1

# 40X Segmentation
run-segmentation data/phenotyping/Well1_Point1_0495.nd2 --magnification 40x --output-dir results/segmentation_40x/Well1

# Mapping between 10X and 40X
run-mapping --seg-10x-dir results/segmentation_10x --seg-40x-dir results/segmentation_40x --output-dir results/mapping --wells Well1

# Genotyping
run-genotyping data/genotyping/cycle_1/Well1_Point1_0034.nd2 --segmentation-dir results/segmentation_10x --barcode-library data/barcodes.csv --wells Well1 --output-dir results/genotyping

# Phenotyping
run-phenotyping data/phenotyping/Well1_Point1_0495.nd2 --segmentation-dir results/segmentation_40x --channels DAPI,mClov3,TMR --wells Well1 --output-dir results/phenotyping

# Album Creation
create-albums --phenotyping-dir results/phenotyping --wells Well1 --output-dir results/albums
```

### Using as a Python package

```python
# Import the package
import imageanalysis

# Create and run segmentation pipeline
from imageanalysis.core.segmentation import Segmentation10XPipeline, Segmentation40XPipeline

# 10X segmentation
pipeline_10x = Segmentation10XPipeline({
    'input_file': 'data/genotyping/cycle_1/Well1_Point1_0034.nd2',
    'output_dir': 'results/segmentation_10x',
    'nuclear_channel': 0,
    'cell_channel': 1
})
results_10x = pipeline_10x.run()

# 40X segmentation
pipeline_40x = Segmentation40XPipeline({
    'input_file': 'data/phenotyping/Well1_Point1_0495.nd2',
    'output_dir': 'results/segmentation_40x', 
    'nuclear_channel': 0,
    'cell_channel': 1
})
results_40x = pipeline_40x.run()

# Mapping between 10X and 40X
from imageanalysis.core.mapping import MappingPipeline

mapping = MappingPipeline(
    seg_10x_dir='results/segmentation_10x',
    seg_40x_dir='results/segmentation_40x',
    output_dir='results/mapping'
)
mapping_results = mapping.run(wells=['Well1'])

# Get the transform for a specific well
transform = mapping_results['Well1'].transform
```

## Testing

Run tests using the test scripts in the `tests` directory:

```bash
# Run basic tests
python -m tests.test_basic

# Run synthetic pipeline tests
python -m tests.synthetic_tests.test_synthetic_pipeline

# Run full pipeline tests with real data
python -m tests.full_pipeline_tests.test_real_pipeline

# Run all tests
python -m tests.run_all_tests
```

## Documentation

Additional documentation:
- [Installation Guide](INSTALLATION.md)
- [Migration Guide](MIGRATION_GUIDE.md) - For moving code from original to new structure
- [Function Reference](FunctionReference.md)
- [Package Structure](PACKAGE_STRUCTURE.md)