"""Base class for segmentation pipelines."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import json
import numpy as np

from ImageAnalysis.core.pipeline import Pipeline
from ImageAnalysis.utils.io import ImageLoader

class SegmentationPipeline(Pipeline):
    """Base class for image segmentation pipelines.
    
    This class extends the base Pipeline class with functionality specific
    to image segmentation tasks.
    """
    
    def __init__(
        self,
        input_file: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        config_file: Optional[Union[str, Path]] = None,
        nuclei_diameter: int = 30,
        cell_diameter: int = 60,
        threshold_std: float = 3.0,
        min_cell_size: int = 100,
        max_cell_size: Optional[int] = None
    ):
        """Initialize the segmentation pipeline.
        
        Args:
            input_file: Path to input ND2 file
            output_dir: Optional output directory (defaults to input_dir/segmentation)
            config_file: Optional path to JSON configuration file
            nuclei_diameter: Expected diameter of nuclei in pixels
            cell_diameter: Expected diameter of cells in pixels
            threshold_std: Number of standard deviations for thresholding
            min_cell_size: Minimum cell size in pixels
            max_cell_size: Optional maximum cell size in pixels
        """
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(input_file), 'segmentation')
            
        super().__init__(
            input_file=input_file,
            output_dir=output_dir,
            config_file=config_file
        )
        
        # Store parameters
        self.nuclei_diameter = nuclei_diameter
        self.cell_diameter = cell_diameter
        self.threshold_std = threshold_std
        self.min_cell_size = min_cell_size
        self.max_cell_size = max_cell_size
        
        # Initialize image loader
        self.image_loader = ImageLoader(self.input_file)
        
    def validate_inputs(self) -> None:
        """Validate pipeline inputs.
        
        Extends base validation with segmentation-specific checks.
        
        Raises:
            ValueError: If inputs are invalid
        """
        super().validate_inputs()
        
        # Validate parameters
        if self.nuclei_diameter <= 0:
            raise ValueError("Nuclei diameter must be positive")
            
        if self.cell_diameter <= 0:
            raise ValueError("Cell diameter must be positive")
            
        if self.cell_diameter <= self.nuclei_diameter:
            raise ValueError("Cell diameter must be larger than nuclei diameter")
            
        if self.threshold_std <= 0:
            raise ValueError("Threshold standard deviation must be positive")
            
        if self.min_cell_size <= 0:
            raise ValueError("Minimum cell size must be positive")
            
        if self.max_cell_size is not None:
            if self.max_cell_size <= self.min_cell_size:
                raise ValueError("Maximum cell size must be larger than minimum")
                
    def get_wells(self) -> List[str]:
        """Get list of wells in the input file.
        
        Returns:
            List of well identifiers
        """
        return self.image_loader.get_wells()
        
    def get_tiles_for_well(self, well_id: str) -> List[str]:
        """Get list of tiles for a specific well.
        
        Args:
            well_id: Well identifier
            
        Returns:
            List of tile identifiers
        """
        return self.image_loader.get_tiles_for_well(well_id)
        
    def segment_tile(
        self,
        well_id: str,
        tile_id: str
    ) -> Dict[str, Any]:
        """Segment a single tile.
        
        This method should be implemented by subclasses to perform the actual
        segmentation. The base implementation raises NotImplementedError.
        
        Args:
            well_id: Well identifier
            tile_id: Tile identifier
            
        Returns:
            Dictionary containing segmentation results
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Segmentation not implemented")
        
    def save_tile_results(
        self,
        well_id: str,
        tile_id: str,
        results: Dict[str, Any]
    ) -> None:
        """Save segmentation results for a tile.
        
        Args:
            well_id: Well identifier
            tile_id: Tile identifier
            results: Dictionary of segmentation results
        """
        # Create well directory
        well_dir = os.path.join(self.output_dir, well_id)
        os.makedirs(well_dir, exist_ok=True)
        
        # Save tile results
        tile_file = os.path.join(well_dir, f"{tile_id}.json")
        with open(tile_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Update well summary
        summary_file = os.path.join(well_dir, "summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        else:
            summary = {
                'total_cells': 0,
                'total_nuclei': 0,
                'tiles': []
            }
            
        summary['total_cells'] += len(results.get('cell_props', []))
        summary['total_nuclei'] += len(results.get('nuclear_props', []))
        if tile_id not in summary['tiles']:
            summary['tiles'].append(tile_id)
            
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
    def run(self, wells: Optional[List[str]] = None) -> None:
        """Run the segmentation pipeline.
        
        Args:
            wells: Optional list of wells to process (defaults to all wells)
        """
        # Validate inputs
        self.validate_inputs()
        
        # Get wells to process
        if wells is None:
            wells = self.get_wells()
            
        # Process each well
        for well_id in wells:
            self.logger.info(f"Processing well: {well_id}")
            
            # Get tiles for well
            tiles = self.get_tiles_for_well(well_id)
            
            # Process each tile
            for tile_id in tiles:
                self.logger.info(f"Processing tile: {tile_id}")
                
                try:
                    # Segment tile
                    results = self.segment_tile(well_id, tile_id)
                    
                    # Save results
                    self.save_tile_results(well_id, tile_id, results)
                    
                except Exception as e:
                    self.logger.error(
                        f"Error processing tile {tile_id} in well {well_id}: {e}"
                    )
                    continue
                    
        self.logger.info("Segmentation completed successfully") 