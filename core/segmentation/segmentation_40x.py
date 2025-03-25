"""40X magnification segmentation pipeline implementation."""

from typing import Any, Dict, List, Optional, Tuple
import os
import numpy as np
from skimage import filters, morphology, measure

from .base import SegmentationPipeline
from .cell_segmentation import CellSegmentation
from ...utils.io import ImageLoader
from ...config.settings import NUCLEI_DIAMETER_40X, CELL_DIAMETER_40X

class Segmentation40XPipeline(SegmentationPipeline):
    """Pipeline for segmenting 40X magnification images."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the 40X segmentation pipeline.
        
        Uses default parameters optimized for 40X magnification:
        - nuclei_diameter: 120 pixels
        - cell_diameter: 240 pixels
        - threshold_std: 2.5
        - min_cell_size: 1600 pixels
        - max_cell_size: 8000 pixels
        """
        # Set 40X-specific defaults
        kwargs.setdefault('nuclei_diameter', 120)
        kwargs.setdefault('cell_diameter', 240)
        kwargs.setdefault('threshold_std', 2.5)
        kwargs.setdefault('min_cell_size', 1600)
        kwargs.setdefault('max_cell_size', 8000)
        
        super().__init__(*args, **kwargs)
        
        # Initialize segmentation module
        self.segmenter = CellSegmentation(
            nuclei_diameter=self.nuclei_diameter,
            cell_diameter=self.cell_diameter,
            threshold_std=self.threshold_std,
            min_cell_size=self.min_cell_size,
            max_cell_size=self.max_cell_size
        )
        self.image_loader = ImageLoader()
        
        # Get input file path
        self.input_file = self.config.get('input_file')
        if not self.input_file:
            raise ValueError("Input file path must be provided in config")
            
    def validate_inputs(self) -> bool:
        """Validate pipeline inputs and configuration.
        
        Returns:
            True if validation passes
        """
        if not os.path.exists(self.input_file):
            self.logger.error(f"Input file not found: {self.input_file}")
            return False
            
        if not self.config.get('wells'):
            self.logger.warning("No wells specified in config, will process all wells")
            # Get all wells from the file
            wells_tiles = self.image_loader.get_wells_and_tiles(self.input_file)
            self.config['wells'] = list(wells_tiles.keys())
            
        return True
        
    def get_tiles_for_well(self, well_id: str) -> List[int]:
        """Get list of tile indices for a given well.
        
        Args:
            well_id: Well identifier
            
        Returns:
            List of tile indices
        """
        wells_tiles = self.image_loader.get_wells_and_tiles(self.input_file)
        return wells_tiles.get(well_id, [])
        
    def segment_tile(
        self,
        well_id: str,
        tile_id: str
    ) -> Dict[str, Any]:
        """Segment a single tile.
        
        Args:
            well_id: Well identifier
            tile_id: Tile identifier
            
        Returns:
            Dictionary containing segmentation results
        """
        # Load image data
        image = self.image_loader.load_nd2_image(well_id, tile_id)
        
        # Get DAPI channel (assumed to be first channel)
        dapi = image[..., 0]
        
        # Segment nuclei
        nuclear_mask = self.segmenter.segment_nuclei(dapi)
        
        # Segment cells using nuclear guidance
        cell_mask = self.segmenter.segment_cells(
            image,
            nuclear_mask=nuclear_mask
        )
        
        # Extract properties
        nuclear_props = self.segmenter.extract_nuclear_properties(
            nuclear_mask,
            image
        )
        
        cell_props = self.segmenter.extract_cell_properties(
            cell_mask,
            nuclear_mask,
            image
        )
        
        # Return results
        return {
            'nuclear_mask': nuclear_mask.tolist(),
            'cell_mask': cell_mask.tolist(),
            'nuclear_props': nuclear_props,
            'cell_props': cell_props
        }
        
    def _preprocess_40x_image(self, image: np.ndarray) -> np.ndarray:
        """Apply 40X-specific preprocessing to the image.
        
        Args:
            image: Input image array
            
        Returns:
            Preprocessed image
        """
        from skimage import restoration, exposure
        
        # Denoise the image using non-local means
        if len(image.shape) > 2:
            denoised = np.zeros_like(image)
            for channel in range(image.shape[-1]):
                denoised[..., channel] = restoration.denoise_nl_means(
                    image[..., channel],
                    patch_size=5,
                    patch_distance=6,
                    h=0.08
                )
        else:
            denoised = restoration.denoise_nl_means(
                image,
                patch_size=5,
                patch_distance=6,
                h=0.08
            )
            
        # Enhance contrast
        enhanced = exposure.equalize_adapthist(denoised)
        
        return enhanced
        
    def run_for_well(self, well_id: str) -> Dict[str, Any]:
        """Run segmentation pipeline for a single well.
        
        Args:
            well_id: Well identifier
            
        Returns:
            Dictionary containing segmentation results
        """
        self.logger.info(f"Processing well {well_id}")
        results = {}
        
        tiles = self.get_tiles_for_well(well_id)
        if not tiles:
            self.logger.warning(f"No tiles found for well {well_id}")
            return results
            
        for tile_id in tiles:
            self.logger.info(f"Processing tile {tile_id}")
            try:
                results[tile_id] = self.segment_tile(well_id, tile_id)
                
            except Exception as e:
                self.logger.error(f"Error processing tile {tile_id}: {str(e)}")
                continue
                
        return results 