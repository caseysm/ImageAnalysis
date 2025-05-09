# ImageAnalysis Package Installation Guide

After updating the codebase to use proper package structure, follow these steps to install and use the package.

## Installation

### Install in Development Mode

For active development, install the package in development mode:

```bash
# Navigate to the package root directory
cd /home/casey/Desktop/ShalemLab/ImageAnalysis

# Install in development mode
pip install -e .
```

This creates a link to your source code, allowing changes to be immediately available without reinstalling.

### Regular Installation

For normal use, install as a regular package:

```bash
# Navigate to the package root directory 
cd /home/casey/Desktop/ShalemLab/ImageAnalysis

# Install the package
pip install .
```

## Testing the Installation

After installation, verify that the package works correctly:

```bash
# Run the simplified test
python tests/test_simplified.py

# Run a specific command-line script
python -m ImageAnalysis.bin.run_segmentation --help
```

## Using the Package

### From Python Scripts

```python
# Import components from the package
from ImageAnalysis.core.segmentation import Segmentation10XPipeline
from ImageAnalysis.utils.io import ImageLoader

# Use the components
loader = ImageLoader('/path/to/image.nd2')
pipeline = Segmentation10XPipeline(input_file='/path/to/image.nd2')
```

### From Command Line

After installation, the command-line tools will be available:

```bash
# Run segmentation
run-segmentation /path/to/image.nd2 --magnification 10x

# Run genotyping
run-genotyping /path/to/image.nd2 --segmentation-dir /path/to/segmentation
```

## Troubleshooting

If you encounter import errors after installation:

1. Verify your Python path includes the installation directory:
   ```python
   import sys
   print(sys.path)
   ```

2. Ensure there are no conflicting packages installed:
   ```bash
   pip list | grep ImageAnalysis
   ```

3. Try reinstalling with the `--force-reinstall` flag:
   ```bash
   pip install -e . --force-reinstall
   ```