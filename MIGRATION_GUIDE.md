# Migration Guide

This guide outlines the steps needed to migrate existing code from the original package structure to the new structure.

## Migration Steps

### 1. Package Installation

Before migration, ensure the package is properly installed in your environment:

```bash
# Navigate to the package directory
cd /path/to/ImageAnalysis

# Install in development mode
pip install -e .
```

### 2. Import Changes

#### Old Import Style

```python
# Old import style (relative imports)
from ...utils.io import ImageLoader
from ...core.pipeline import Pipeline
from ..segmentation.base import SegmentationPipeline
```

#### New Import Style

```python
# New import style (absolute imports)
from imageanalysis.utils.io import ImageLoader
from imageanalysis.core.pipeline import Pipeline
from imageanalysis.core.segmentation.base import SegmentationPipeline
```

### 3. Script Updates

#### Command-line Scripts

The bin scripts have been updated to use the new import structure:

```python
# Old style
from ...core.segmentation import SegmentationPipeline

# New style
from imageanalysis.core.segmentation import SegmentationPipeline
```

#### Test Scripts

Tests should be updated to use the new import structure:

```python
# Old style
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import ImageLoader

# New style
from imageanalysis.utils.io import ImageLoader
```

### 4. Moving Existing Code

To migrate existing code into the new structure:

1. Identify the module's place in the new structure
2. Copy the file to the appropriate directory in the `imageanalysis` package
3. Update imports in the file to use absolute imports
4. Update references to the file in other modules

### 5. Updating Tests

Update test files to use the new import structure:

```python
# Old style
from tests.utils import create_test_data

# New style
from tests.utils import create_test_data  # Keep test-specific imports as is
from imageanalysis.utils.io import ImageLoader  # Update package imports
```

## Functionality Changes

The core functionality remains the same, but is now better organized:

- Pipeline base classes in `imageanalysis.core`
- Specific pipeline implementations in their respective modules
- Utility functions in `imageanalysis.utils`
- Configuration settings in `imageanalysis.config`
- Command-line scripts in `imageanalysis.bin`

## Key Structural Changes

The package has been restructured to follow Python packaging best practices:

1. **Lowercase package name**: Changed from `ImageAnalysis` to `imageanalysis` following Python conventions
2. **Proper package structure**: Created a dedicated `imageanalysis` directory containing all package code
3. **Improved setup.py**: Updated to use `find_packages()` and properly define entry points
4. **Clear module boundaries**: Each module has well-defined responsibilities and interfaces

## Running Tests

To verify your migration was successful, run the test scripts:

```bash
# Test package imports
python test_new_structure.py

# Run unit tests (once migrated)
pytest tests/
```