"""Standard implementation of the phenotyping pipeline."""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from skimage import measure, filters

from ImageAnalysis.core.phenotyping.base import PhenotypingPipeline
from ImageAnalysis.core.phenotyping.metrics import calculate_all_metrics
from ImageAnalysis.utils.io import ImageLoader

class StandardPhenotypingPipeline(PhenotypingPipeline):
    """Standard implementation of phenotype measurement pipeline.
    
    This pipeline implements common phenotype measurements including:
    - Area and shape metrics for cells and nuclei
    - Intensity measurements across channels
    - Texture analysis
    - Spatial/location measurements
    """
    
    def __init__(
        self,
        input_file: Union[str, Path],
        segmentation_dir: Union[str, Path],
        channels: List[str],
        metrics: List[str] = ['area', 'intensity', 'shape', 'texture', 'location'],
        genotyping_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        config_file: Optional[Union[str, Path]] = None,
        pixel_size: float = 1.0,
        min_cell_size: int = 100,
        max_cell_size: int = 10000,
        background_percentile: float = 1.0
    ):
        """Initialize the standard phenotyping pipeline.
        
        Args:
            input_file: Path to input ND2 file
            segmentation_dir: Directory containing segmentation results
            channels: List of channel names in order they appear in ND2 file
            metrics: List of metrics to calculate
            genotyping_dir: Optional directory containing genotyping results
            output_dir: Optional output directory
            config_file: Optional path to JSON configuration file
            pixel_size: Physical size of each pixel in microns
            min_cell_size: Minimum cell size in pixels
            max_cell_size: Maximum cell size in pixels
            background_percentile: Percentile for background estimation
        """
        super().__init__(
            input_file=input_file,
            segmentation_dir=segmentation_dir,
            channels=channels,
            metrics=metrics,
            genotyping_dir=genotyping_dir,
            output_dir=output_dir,
            config_file=config_file,
            pixel_size=pixel_size
        )
        
        # Store additional parameters
        self.min_cell_size = min_cell_size
        self.max_cell_size = max_cell_size
        self.background_percentile = background_percentile
        
    def estimate_background(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> float:
        """Estimate background intensity from non-cell regions.
        
        Args:
            image: Intensity image
            mask: Binary mask of all cells
            
        Returns:
            Estimated background intensity
        """
        # Get intensities from non-cell regions
        background_intensities = image[~mask]
        
        # Calculate background as low percentile
        if len(background_intensities) > 0:
            return np.percentile(background_intensities, self.background_percentile)
        else:
            return 0.0
            
    def filter_cells(
        self,
        cell_mask: np.ndarray,
        nuclei_mask: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Filter cells based on size and quality criteria.
        
        Args:
            cell_mask: Label mask for cells
            nuclei_mask: Label mask for nuclei
            
        Returns:
            Tuple of (filtered cell masks, filtered nuclei masks)
        """
        # Get cell properties
        cell_props = measure.regionprops(cell_mask)
        nuclei_props = measure.regionprops(nuclei_mask)
        
        # Filter based on size
        valid_cells = []
        valid_nuclei = []
        
        for cell, nucleus in zip(cell_props, nuclei_props):
            if (self.min_cell_size <= cell.area <= self.max_cell_size and
                nucleus.area > 0):
                
                # Create binary masks for this cell/nucleus
                cell_binary = cell_mask == cell.label
                nuclei_binary = nuclei_mask == nucleus.label
                
                valid_cells.append(cell_binary)
                valid_nuclei.append(nuclei_binary)
                
        return valid_cells, valid_nuclei
        
    def get_well_center(
        self,
        mask_shape: Tuple[int, int]
    ) -> Tuple[float, float]:
        """Calculate the center coordinates of the well/image.
        
        Args:
            mask_shape: Shape of the mask array (height, width)
            
        Returns:
            Tuple of (center_y, center_x) coordinates
        """
        height, width = mask_shape
        return (height / 2, width / 2)
        
    def process_tile(
        self,
        well_id: str,
        tile_id: str
    ) -> Dict[str, Any]:
        """Process a single tile.
        
        Args:
            well_id: Well identifier
            tile_id: Tile identifier
            
        Returns:
            Dictionary containing phenotyping results
            
        Raises:
            FileNotFoundError: If required files are missing
            ValueError: If processing fails
        """
        # Load segmentation results
        seg_results = self.load_segmentation_results(well_id, tile_id)
        
        # Load masks
        cell_mask = np.load(os.path.join(
            self.segmentation_dir, well_id, seg_results['cell_mask_file']
        ))
        nuclei_mask = np.load(os.path.join(
            self.segmentation_dir, well_id, seg_results['nuclei_mask_file']
        ))
        
        # Load image data for this tile
        image_data = self.image_loader.load_tile(well_id, tile_id)
        
        # Organize channels
        channels = {}
        for i, channel in enumerate(self.channels):
            channels[channel] = image_data[..., i]
            
        # Get well center for location calculations
        well_center = self.get_well_center(cell_mask.shape)
        
        # Filter cells
        valid_cells, valid_nuclei = self.filter_cells(cell_mask, nuclei_mask)
        
        # Calculate background intensities
        background_intensities = {}
        combined_mask = np.zeros_like(cell_mask, dtype=bool)
        for cell_mask in valid_cells:
            combined_mask |= cell_mask
            
        for channel, image in channels.items():
            background_intensities[channel] = self.estimate_background(
                image, combined_mask
            )
            
        # Process each cell
        results = {
            'well_id': well_id,
            'tile_id': tile_id,
            'total_cells': len(valid_cells),
            'cells': []
        }
        
        # Load genotyping results if available
        genotyping_results = self.load_genotyping_results(well_id, tile_id)
        
        # Process each cell
        for i, (cell_mask, nuclei_mask) in enumerate(zip(valid_cells, valid_nuclei)):
            try:
                # Calculate all metrics
                cell_metrics = calculate_all_metrics(
                    cell_mask=cell_mask,
                    nuclei_mask=nuclei_mask,
                    channels=channels,
                    pixel_size=self.pixel_size,
                    well_center=well_center,
                    background_intensities=background_intensities
                )
                
                # Add cell index and location info
                cell_data = {
                    'cell_id': f"{well_id}_{tile_id}_{i}",
                    'metrics': cell_metrics
                }
                
                # Add genotyping info if available
                if genotyping_results and str(i) in genotyping_results.get('cells', {}):
                    cell_data['genotype'] = genotyping_results['cells'][str(i)]
                    
                results['cells'].append(cell_data)
                
            except Exception as e:
                self.logger.warning(
                    f"Failed to process cell {i} in tile {tile_id}: {str(e)}"
                )
                continue
                
        return results
        
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save phenotyping results.
        
        Args:
            results: Dictionary of phenotyping results
        """
        # Create output directory for well
        well_dir = os.path.join(self.output_dir, results['well_id'])
        os.makedirs(well_dir, exist_ok=True)
        
        # Save results for tile
        output_file = os.path.join(
            well_dir,
            f"{results['tile_id']}.json"
        )
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Update summary file
        summary_file = os.path.join(well_dir, 'summary.json')
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        else:
            summary = {'total_cells': 0, 'tiles': []}
            
        summary['total_cells'] += len(results['cells'])
        summary['tiles'].append(results['tile_id'])
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2) 