# Package Structure Guide

This document outlines the changes made to the package structure to ensure proper importing and usage.

## Changes Made

1. Created a proper Python package structure:
   - Added a lowercase `imageanalysis` package directory
   - Moved core functionality inside this directory
   - Updated imports to use absolute paths from the package root

2. Fixed the setup.py file:
   - Changed the package name to lowercase `imageanalysis` (Python convention)
   - Used setuptools `find_packages()` to automatically find all package modules
   - Configured entry points for command-line scripts

3. Updated import statements:
   - Changed relative imports like `from ...utils.io import ImageLoader` to absolute imports like `from imageanalysis.utils.io import ImageLoader`
   - Removed references to the package from outside its own namespace

4. Created proper `__init__.py` files for each module:
   - Added documentation strings
   - Imported key components for easy access

## Module Organization

### Core Module

The `core` module contains the main pipeline components:

- `pipeline.py` - Base class for all pipeline components
- `segmentation/` - Image segmentation functionality
- `genotyping/` - Cell genotyping functionality
- `phenotyping/` - Cell phenotyping functionality
- `mapping/` - Image registration and mapping
- `visualization/` - Visualization tools

### Utils Module

The `utils` module contains utility functions and classes:

- `io.py` - Input/output utilities
- `logging.py` - Logging utilities

### Config Module

The `config` module contains configuration settings:

- `settings.py` - Default settings and configurations

### Bin Module

The `bin` module contains command-line scripts:

- `run_segmentation.py` - Run the segmentation pipeline
- `run_genotyping.py` - Run the genotyping pipeline
- `run_phenotyping.py` - Run the phenotyping pipeline
- `create_albums.py` - Create cell albums for visualization

## Installation and Usage

### Installation

Install the package in development mode:

```bash
pip install -e .
```

This will install the package and make it available for importing in your environment.

### Importing

```python
# Import the package
import imageanalysis

# Import specific modules
from imageanalysis.core.segmentation import SegmentationPipeline
from imageanalysis.utils.io import ImageLoader
```

### Command Line Tools

The package provides several command-line tools:

```bash
# Run segmentation pipeline
run-segmentation path/to/image.nd2 -o path/to/output -m 10x
```

## Testing

To test the package structure:

```bash
python test_new_structure.py
```

This script verifies that the package can be imported correctly.