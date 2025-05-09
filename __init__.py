"""ImageAnalysis - A package for microscopy image analysis and processing.

This package provides tools for segmentation, genotyping, phenotyping,
and visualization of microscopy images.
"""

__version__ = "0.1.0"

# Core modules
from . import core
from . import utils
from . import config

# Expose key classes and functions for easy access
from .core.pipeline import Pipeline
from .core.segmentation import SegmentationPipeline, Segmentation10XPipeline, Segmentation40XPipeline
from .core.genotyping import GenotypingPipeline, StandardGenotypingPipeline
from .core.phenotyping import PhenotypingPipeline, StandardPhenotypingPipeline
from .utils.io import ImageLoader, save_image