"""Cell segmentation modules."""

from imageanalysis.core.segmentation.segmentation_10x import Segmentation10XPipeline
from imageanalysis.core.segmentation.segmentation_40x import Segmentation40XPipeline

__all__ = ['Segmentation10XPipeline', 'Segmentation40XPipeline']
