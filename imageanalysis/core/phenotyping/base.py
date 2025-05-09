"""Base class for phenotyping pipelines."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import json
import numpy as np

from ImageAnalysis.core.pipeline import Pipeline
from ImageAnalysis.utils.io import ImageLoader

class PhenotypingPipeline(Pipeline):
    """Base class for phenotyping pipelines.
    
    This class extends the base Pipeline class with functionality specific
    to phenotype measurement tasks.
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
        pixel_size: float = 1.0
    ):
        """Initialize the phenotyping pipeline.
        
        Args:
            input_file: Path to input ND2 file
            segmentation_dir: Directory containing segmentation results
            channels: List of channel names in order they appear in ND2 file
            metrics: List of metrics to calculate
            genotyping_dir: Optional directory containing genotyping results
            output_dir: Optional output directory (defaults to segmentation_dir/phenotyping)
            config_file: Optional path to JSON configuration file
            pixel_size: Physical size of each pixel in microns
        """
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = os.path.join(segmentation_dir, 'phenotyping')
            
        super().__init__(
            input_file=input_file,
            output_dir=output_dir,
            config_file=config_file
        )
        
        # Store parameters
        self.segmentation_dir = Path(segmentation_dir)
        self.genotyping_dir = Path(genotyping_dir) if genotyping_dir else None
        self.channels = channels
        self.metrics = metrics
        self.pixel_size = pixel_size
        
        # Initialize image loader
        self.image_loader = ImageLoader(self.input_file)
        
    def validate_inputs(self) -> None:
        """Validate pipeline inputs.
        
        Extends base validation with phenotyping-specific checks.
        
        Raises:
            ValueError: If inputs are invalid
        """
        super().validate_inputs()
        
        # Check segmentation directory exists
        if not self.segmentation_dir.exists():
            raise ValueError(f"Segmentation directory not found: {self.segmentation_dir}")
            
        # Check genotyping directory if provided
        if self.genotyping_dir and not self.genotyping_dir.exists():
            raise ValueError(f"Genotyping directory not found: {self.genotyping_dir}")
            
        # Validate channels
        if not self.channels:
            raise ValueError("At least one channel must be specified")
            
        # Validate metrics
        valid_metrics = ['area', 'intensity', 'shape', 'texture', 'location']
        invalid_metrics = [m for m in self.metrics if m not in valid_metrics]
        if invalid_metrics:
            raise ValueError(f"Invalid metrics specified: {invalid_metrics}")
            
        # Validate pixel size
        if self.pixel_size <= 0:
            raise ValueError("Pixel size must be positive")
            
    def load_segmentation_results(
        self,
        well_id: str,
        tile_id: str
    ) -> Dict[str, Any]:
        """Load segmentation results for a tile.
        
        Args:
            well_id: Well identifier
            tile_id: Tile identifier
            
        Returns:
            Dictionary containing segmentation results
            
        Raises:
            FileNotFoundError: If segmentation results don't exist
        """
        results_file = os.path.join(
            self.segmentation_dir,
            well_id,
            f"{tile_id}.json"
        )
        
        if not os.path.exists(results_file):
            raise FileNotFoundError(
                f"Segmentation results not found: {results_file}"
            )
            
        with open(results_file, 'r') as f:
            return json.load(f)
            
    def load_genotyping_results(
        self,
        well_id: str,
        tile_id: str
    ) -> Optional[Dict[str, Any]]:
        """Load genotyping results for a tile if available.
        
        Args:
            well_id: Well identifier
            tile_id: Tile identifier
            
        Returns:
            Dictionary containing genotyping results, or None if not available
        """
        if not self.genotyping_dir:
            return None
            
        results_file = os.path.join(
            self.genotyping_dir,
            well_id,
            f"{tile_id}.json"
        )
        
        if not os.path.exists(results_file):
            return None
            
        with open(results_file, 'r') as f:
            return json.load(f)
            
    def save_results(
        self,
        well_id: str,
        tile_id: str,
        results: Dict[str, Any]
    ) -> None:
        """Save phenotyping results for a tile.
        
        Args:
            well_id: Well identifier
            tile_id: Tile identifier
            results: Dictionary of phenotyping results
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
                'total_metrics': 0,
                'tiles': []
            }
            
        summary['total_cells'] += len(results.get('cells', []))
        summary['total_metrics'] = len(self.metrics)
        if tile_id not in summary['tiles']:
            summary['tiles'].append(tile_id)
            
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
    def process_tile(
        self,
        well_id: str,
        tile_id: str
    ) -> Dict[str, Any]:
        """Process a single tile.
        
        This method should be implemented by subclasses to perform the actual
        phenotype measurements. The base implementation raises NotImplementedError.
        
        Args:
            well_id: Well identifier
            tile_id: Tile identifier
            
        Returns:
            Dictionary containing phenotyping results
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Phenotype measurement not implemented")
        
    def run(self, wells: Optional[List[str]] = None) -> None:
        """Run the phenotyping pipeline.
        
        Args:
            wells: Optional list of wells to process (defaults to all wells)
        """
        # Validate inputs
        self.validate_inputs()
        
        # Get wells to process
        if wells is None:
            wells = [d.name for d in self.segmentation_dir.iterdir()
                    if d.is_dir()]
            
        # Process each well
        for well_id in wells:
            self.logger.info(f"Processing well: {well_id}")
            
            # Get tiles for well
            tiles = [f.stem for f in (self.segmentation_dir / well_id).glob('*.json')
                    if f.name != 'summary.json']
            
            # Process each tile
            for tile_id in tiles:
                self.logger.info(f"Processing tile: {tile_id}")
                
                try:
                    # Process tile
                    results = self.process_tile(well_id, tile_id)
                    
                    # Save results
                    self.save_results(well_id, tile_id, results)
                    
                except Exception as e:
                    self.logger.error(
                        f"Error processing tile {tile_id} in well {well_id}: {e}"
                    )
                    continue
                    
        self.logger.info("Phenotype analysis completed successfully") 