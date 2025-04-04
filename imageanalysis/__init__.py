"""ImageAnalysis package for image processing and analysis.

This package provides a set of tools for analyzing microscopy images,
including segmentation, genotyping, phenotyping, and visualization.
"""

__version__ = "0.2.0"

# Import key components for easy access
from imageanalysis.core.pipeline import Pipeline
from imageanalysis.core.segmentation import Segmentation10XPipeline, Segmentation40XPipeline
from imageanalysis.core.genotyping.pipeline import StandardGenotypingPipeline
from imageanalysis.core.phenotyping.pipeline import PhenotypingPipeline
from imageanalysis.core.mapping.pipeline import MappingPipeline
from imageanalysis.utils.io import ImageLoader
from imageanalysis.utils.logging import setup_logger
