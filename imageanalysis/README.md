# imageanalysis

A Python package for image analysis pipelines in the Shalem Lab.

## Overview

This package provides tools for image processing, cellular segmentation, genotyping, and phenotyping of microscopy data. It is organized into modules that can be run either independently or as part of a complete pipeline.

## Package Structure

```
imageanalysis/           # Main package directory
├── __init__.py          # Package initialization
├── bin/                 # Command-line scripts
│   ├── run_segmentation.py   # Segmentation script
│   ├── run_genotyping.py     # Genotyping script
│   ├── run_phenotyping.py    # Phenotyping script
│   └── create_albums.py      # Album creation script
├── core/                # Core functionality
│   ├── pipeline.py      # Base pipeline class
│   ├── segmentation/    # Cell segmentation
│   ├── genotyping/      # Cell genotyping
│   ├── phenotyping/     # Cell phenotyping
│   ├── mapping/         # Image registration
│   └── visualization/   # Visualization tools
├── utils/               # Utility functions
│   ├── io.py            # I/O utilities
│   └── logging.py       # Logging utilities
└── config/              # Configuration
    └── settings.py      # Default settings
```

## Installation

Install the package in development mode:

```bash
pip install -e .
```

## Usage

```python
# Import the package
import imageanalysis

# Use specific components
from imageanalysis.core.segmentation import Segmentation10XPipeline
from imageanalysis.utils.io import ImageLoader

# Create a pipeline
pipeline = Segmentation10XPipeline(config={
    'input_file': 'data/image.nd2',
    'output_dir': 'results/segmentation'
})

# Run the pipeline
results = pipeline.run()
```

## Command-line Tools

The package provides command-line tools for running pipelines:

```bash
# Run segmentation
run_segmentation data/image.nd2 --magnification 10x --output-dir results/segmentation

# Run genotyping
run_genotyping data/image.nd2 --segmentation-dir results/segmentation --barcode-library data/barcodes.csv

# Run phenotyping
run_phenotyping data/image.nd2 --segmentation-dir results/segmentation --channels DAPI,GFP,RFP

# Create albums
create_albums --phenotyping-dir results/phenotyping --output-dir results/albums
```