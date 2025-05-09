# Package Structure Fixes

I've updated several key files to fix the package structure issues:

## 1. Fixed `__init__.py` Files

Created proper `__init__.py` files in all package directories:
- Main package `__init__.py`
- Core module `__init__.py` files
- Submodule `__init__.py` files for segmentation, genotyping, etc.

## 2. Fixed Imports

Changed problematic relative imports to absolute imports in key files:
- `core/segmentation/base.py`: Changed `from ...utils.io import ImageLoader` to `from ImageAnalysis.utils.io import ImageLoader`
- `bin/run_segmentation.py`: Removed `sys.path.append` hack and used proper imports

## 3. Added Setup Script

Added a `setup.py` file to make the package properly installable.

## Remaining Tasks

Some files still need to be updated to use absolute imports:

1. Update all remaining scripts in `bin/` directory
2. Update all remaining files in `core/` directories
3. Fix imports in test scripts

## How to Complete the Migration

1. Install the package in development mode:
   ```bash
   cd /home/casey/Desktop/ShalemLab/ImageAnalysis
   pip install -e .
   ```

2. Fix any remaining files using this pattern:
   - Change `from ..some_module import X` to `from ImageAnalysis.some_module import X`
   - Change `from ...utils.io import X` to `from ImageAnalysis.utils.io import X`

3. Test the package by running:
   ```bash
   # Run the commands from the bin directory
   python -m ImageAnalysis.bin.run_segmentation input.nd2 --magnification 10x
   ```

## Additional Tips

- Always use absolute imports (`from ImageAnalysis.module import X`)
- Keep `__init__.py` files up to date when adding new modules
- Use the package name as the root for all imports