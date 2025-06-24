#!/usr/bin/env python3
"""
Simple script to verify imageanalysis package installation and structure.
"""

import sys
import importlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def check_module(module_name):
    """Try to import a module and check if it exists."""
    try:
        module = importlib.import_module(module_name)
        return True
    except ImportError as e:
        logger.error(f"Failed to import {module_name}: {e}")
        return False

def main():
    """Main verification function."""
    logger.info("Verifying imageanalysis package installation...")
    
    # Check main package
    if not check_module('imageanalysis'):
        logger.error("Main package 'imageanalysis' not found")
        return False
    
    # Import and print package version
    import imageanalysis
    logger.info(f"Package version: {imageanalysis.__version__}")
    
    # Check core modules
    core_modules = [
        'imageanalysis.core.pipeline',
        'imageanalysis.core.segmentation.base',
        'imageanalysis.core.segmentation.segmentation_10x',
        'imageanalysis.core.segmentation.segmentation_40x',
        'imageanalysis.core.genotyping.base',
        'imageanalysis.core.genotyping.pipeline',
        'imageanalysis.core.phenotyping.base',
        'imageanalysis.core.phenotyping.pipeline',
        'imageanalysis.core.visualization.albums',
        'imageanalysis.utils.io',
        'imageanalysis.utils.logging',
        'imageanalysis.config.settings'
    ]
    
    success = True
    logger.info("\nChecking core modules:")
    for module in core_modules:
        result = check_module(module)
        if result:
            logger.info(f"✓ {module}")
        else:
            logger.info(f"✗ {module}")
            success = False
    
    # Check for required 3rd party dependencies
    dependencies = [
        'numpy', 'pandas', 'matplotlib', 'cellpose', 'nd2reader', 
        'scipy', 'skimage', 'json', 'pathlib'
    ]
    
    logger.info("\nChecking dependencies:")
    for dep in dependencies:
        result = check_module(dep)
        if result:
            logger.info(f"✓ {dep}")
        else:
            logger.info(f"✗ {dep}")
            success = False
    
    # Verify pipeline class existence
    logger.info("\nVerifying pipeline classes:")
    
    # Import classes directly
    try:
        from imageanalysis.core.segmentation import Segmentation10XPipeline, Segmentation40XPipeline
        from imageanalysis.core.genotyping.pipeline import StandardGenotypingPipeline
        from imageanalysis.core.phenotyping.pipeline import PhenotypingPipeline
        logger.info("✓ All main pipeline classes available")
    except ImportError as e:
        logger.error(f"Failed to import pipeline classes: {e}")
        success = False
    
    if success:
        logger.info("\n✓ Package verification successful!")
        return 0
    else:
        logger.error("\n✗ Package verification failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())