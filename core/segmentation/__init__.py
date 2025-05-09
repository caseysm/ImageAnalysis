"""Segmentation modules for microscopy image analysis.

This package contains implementations of image segmentation algorithms
specifically designed for microscopy images.
"""

from .base import SegmentationPipeline
from .segmentation_10x import Segmentation10XPipeline
from .segmentation_40x import Segmentation40XPipeline