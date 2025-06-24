#!/usr/bin/env python3
"""Test script to verify the package can be imported correctly."""

import importlib.util
import sys

def test_imageanalysis_package():
    """Test that the imageanalysis package can be imported correctly."""
    try:
        import imageanalysis
        print(f"✅ Successfully imported imageanalysis package (version {imageanalysis.__version__})")
        
        # Try importing specific modules
        from imageanalysis.core.segmentation import Segmentation10XPipeline
        print("✅ Successfully imported segmentation module")
        
        # Import specific modules without importing from the package directly
        import imageanalysis.core.genotyping.pipeline
        print("✅ Successfully imported genotyping module")
        
        import imageanalysis.core.phenotyping.pipeline
        print("✅ Successfully imported phenotyping module")
        
        from imageanalysis.utils.io import ImageLoader
        print("✅ Successfully imported utils module")
        
        print("\nAll imports successful! The package is installed correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Error importing package: {e}")
        print("\nThe package may not be installed correctly. Try reinstalling with:")
        print("pip install -e .")
        return False

if __name__ == "__main__":
    print("Testing imageanalysis package imports...\n")
    success = test_imageanalysis_package()
    sys.exit(0 if success else 1)