#!/usr/bin/env python3
"""
Test script to check different ND2 readers on Apple Silicon.
This will try multiple methods to read ND2 files.
"""

import sys
import os
import numpy as np
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nd2-test")

# Set data directory
DATA_DIR = Path('/Users/csm70/Desktop/Shalem_Lab/ImageAnalysis/Data')

# Find first ND2 file
def find_first_nd2():
    for nd2_file in DATA_DIR.glob("**/*.nd2"):
        return nd2_file
    return None

def test_nd2reader():
    """Test the original nd2reader package."""
    try:
        import nd2reader
        logger.info("Testing nd2reader")
        
        nd2_file = find_first_nd2()
        if not nd2_file:
            logger.error("No ND2 files found")
            return False
        
        logger.info(f"Found ND2 file: {nd2_file}")
        
        try:
            with nd2reader.ND2Reader(str(nd2_file)) as images:
                logger.info(f"Successfully opened ND2 file with nd2reader")
                logger.info(f"Image size: {images.sizes}")
                n_channels = images.sizes.get('c', 1)
                logger.info(f"Number of channels: {n_channels}")
                
                # Extract first channel
                images.default_coords['c'] = 0
                image = images[0]
                logger.info(f"First channel shape: {image.shape}")
                return True
        except Exception as e:
            logger.error(f"Failed to read ND2 file with nd2reader: {e}")
            return False
    except ImportError:
        logger.error("nd2reader is not installed")
        return False

def test_nd2():
    """Test the nd2 package."""
    try:
        import nd2
        logger.info("Testing nd2 package")
        
        nd2_file = find_first_nd2()
        if not nd2_file:
            logger.error("No ND2 files found")
            return False
        
        logger.info(f"Found ND2 file: {nd2_file}")
        
        try:
            with nd2.ND2File(str(nd2_file)) as images:
                logger.info(f"Successfully opened ND2 file with nd2 package")
                logger.info(f"Image attributes: {images.attributes}")
                
                # Get shape information
                if hasattr(images, 'shape'):
                    logger.info(f"Image shape: {images.shape}")
                
                # Try to read image data
                image = images.read_frame(0)
                logger.info(f"First frame shape: {image.shape}")
                return True
        except Exception as e:
            logger.error(f"Failed to read ND2 file with nd2 package: {e}")
            return False
    except ImportError:
        logger.error("nd2 package is not installed")
        return False

def test_aicsimageio():
    """Test the aicsimageio package."""
    try:
        from aicsimageio import AICSImage
        logger.info("Testing aicsimageio")
        
        nd2_file = find_first_nd2()
        if not nd2_file:
            logger.error("No ND2 files found")
            return False
        
        logger.info(f"Found ND2 file: {nd2_file}")
        
        try:
            img = AICSImage(str(nd2_file))
            logger.info(f"Successfully opened ND2 file with aicsimageio")
            logger.info(f"Image shape: {img.shape}")
            
            # Get channel data
            channel_data = img.get_channel_data(0)
            logger.info(f"Channel data shape: {channel_data.shape}")
            return True
        except Exception as e:
            logger.error(f"Failed to read ND2 file with aicsimageio: {e}")
            return False
    except ImportError:
        logger.error("aicsimageio is not installed")
        return False

def test_bioformats():
    """Test bioformats-jar with scyjava."""
    try:
        import bioformats_jar
        import scyjava
        import jpype
        
        logger.info("Testing bioformats_jar")
        
        nd2_file = find_first_nd2()
        if not nd2_file:
            logger.error("No ND2 files found")
            return False
        
        logger.info(f"Found ND2 file: {nd2_file}")
        
        try:
            # Initialize Java VM and BioFormats
            bioformats_jar.init_jars()
            jutil = jpype.JPackage('loci').common.DebugTools
            jutil.setRootLevel("ERROR")
            
            reader = jpype.JPackage('loci').formats.reader.ImageReader()
            logger.info(f"Successfully initialized BioFormats")
            
            # Open ND2 file
            reader.setId(str(nd2_file))
            logger.info(f"Successfully opened ND2 file with BioFormats")
            
            # Get metadata
            meta = jpype.JPackage('loci').formats.MetadataTools.createOMEXMLMetadata()
            reader.setMetadataStore(meta)
            
            # Get image dimensions
            width = reader.getSizeX()
            height = reader.getSizeY()
            logger.info(f"Image dimensions: {width}x{height}")
            
            # Read first plane
            img_bytes = reader.openBytes(0)
            logger.info(f"Successfully read image data: {len(img_bytes)} bytes")
            reader.close()
            return True
        except Exception as e:
            logger.error(f"Failed to read ND2 file with bioformats: {e}")
            return False
    except ImportError as e:
        logger.error(f"bioformats_jar or dependencies not installed: {e}")
        return False

def convert_nd2_to_tiff():
    """Convert ND2 file to TIFF using a working reader."""
    try:
        from aicsimageio import AICSImage
        from tifffile import imsave
        
        logger.info("Converting ND2 to TIFF")
        
        nd2_file = find_first_nd2()
        if not nd2_file:
            logger.error("No ND2 files found")
            return False
        
        logger.info(f"Found ND2 file: {nd2_file}")
        
        try:
            img = AICSImage(str(nd2_file))
            logger.info(f"Successfully opened ND2 file for conversion")
            
            # Convert each channel to a TIFF file
            for c in range(img.dims.C):
                channel_data = img.get_image_data("CYX", C=c)
                channel_data = channel_data[0]  # Get first timepoint
                
                # Create output filename
                output_file = nd2_file.with_suffix(f".c{c}.tiff")
                
                # Save as TIFF
                imsave(output_file, channel_data)
                logger.info(f"Saved channel {c} to {output_file}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to convert ND2 to TIFF: {e}")
            return False
    except ImportError:
        logger.error("aicsimageio or tifffile not installed")
        return False

if __name__ == "__main__":
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    
    # Test all readers
    results = {}
    
    # Test nd2reader (original)
    start = time.time()
    results['nd2reader'] = test_nd2reader()
    logger.info(f"nd2reader test completed in {time.time() - start:.2f} seconds")
    
    # Test nd2 (new package)
    start = time.time()
    results['nd2'] = test_nd2()
    logger.info(f"nd2 test completed in {time.time() - start:.2f} seconds")
    
    # Test aicsimageio
    start = time.time()
    results['aicsimageio'] = test_aicsimageio()
    logger.info(f"aicsimageio test completed in {time.time() - start:.2f} seconds")
    
    # Test bioformats
    start = time.time()
    results['bioformats'] = test_bioformats()
    logger.info(f"bioformats test completed in {time.time() - start:.2f} seconds")
    
    # Print summary
    logger.info("\nResults Summary:")
    for reader, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"{reader}: {status}")
    
    # Try to convert if at least one reader worked
    if any(results.values()):
        logger.info("\nAttempting to convert ND2 to TIFF...")
        convert_result = convert_nd2_to_tiff()
        if convert_result:
            logger.info("Successfully converted ND2 to TIFF")
            logger.info("\nYou can now modify the segmentation pipeline to use TIFF files instead of ND2 files.")
        else:
            logger.info("Failed to convert ND2 to TIFF")
    else:
        logger.error("All ND2 readers failed on this system")