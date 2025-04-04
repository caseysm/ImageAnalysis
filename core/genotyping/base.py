"""Base class for genotyping pipelines."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import json
import numpy as np
import pandas as pd

from ImageAnalysis.core.pipeline import Pipeline
from ImageAnalysis.utils.io import ImageLoader

class GenotypingPipeline(Pipeline):
    """Base class for genotyping pipelines.
    
    This class extends the base Pipeline class with functionality specific
    to genotyping tasks.
    """
    
    def __init__(
        self,
        input_file: Union[str, Path],
        segmentation_dir: Union[str, Path],
        barcode_library: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        config_file: Optional[Union[str, Path]] = None,
        min_quality_score: float = 0.8,
        max_hamming_distance: int = 1
    ):
        """Initialize the genotyping pipeline.
        
        Args:
            input_file: Path to input ND2 file
            segmentation_dir: Directory containing segmentation results
            barcode_library: Path to barcode library CSV file
            output_dir: Optional output directory (defaults to segmentation_dir/genotyping)
            config_file: Optional path to JSON configuration file
            min_quality_score: Minimum quality score for base calls
            max_hamming_distance: Maximum Hamming distance for barcode matching
        """
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = os.path.join(segmentation_dir, 'genotyping')
            
        super().__init__(
            input_file=input_file,
            output_dir=output_dir,
            config_file=config_file
        )
        
        # Store parameters
        self.segmentation_dir = Path(segmentation_dir)
        self.barcode_library = Path(barcode_library)
        self.min_quality_score = min_quality_score
        self.max_hamming_distance = max_hamming_distance
        
        # Initialize image loader
        self.image_loader = ImageLoader(self.input_file)
        
    def validate_inputs(self) -> None:
        """Validate pipeline inputs.
        
        Extends base validation with genotyping-specific checks.
        
        Raises:
            ValueError: If inputs are invalid
        """
        super().validate_inputs()
        
        # Check segmentation directory exists
        if not self.segmentation_dir.exists():
            raise ValueError(f"Segmentation directory not found: {self.segmentation_dir}")
            
        # Check barcode library exists
        if not self.barcode_library.exists():
            raise ValueError(f"Barcode library not found: {self.barcode_library}")
            
        # Validate parameters
        if not 0 <= self.min_quality_score <= 1:
            raise ValueError("Quality score must be between 0 and 1")
            
        if self.max_hamming_distance < 0:
            raise ValueError("Hamming distance must be non-negative")
            
    def load_barcode_library(self) -> pd.DataFrame:
        """Load the barcode library.
        
        Returns:
            DataFrame containing barcode sequences and metadata
        """
        try:
            return pd.read_csv(self.barcode_library)
        except Exception as e:
            raise ValueError(f"Error loading barcode library: {e}")
            
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
            
    def save_results(
        self,
        well_id: str,
        tile_id: str,
        results: Dict[str, Any]
    ) -> None:
        """Save genotyping results for a tile.
        
        Args:
            well_id: Well identifier
            tile_id: Tile identifier
            results: Dictionary of genotyping results
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
                'assigned_cells': 0,
                'tiles': []
            }
            
        summary['total_cells'] += len(results.get('cells', []))
        summary['assigned_cells'] += sum(
            1 for cell in results.get('cells', [])
            if cell.get('barcode') is not None
        )
        if tile_id not in summary['tiles']:
            summary['tiles'].append(tile_id)
            
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
    def genotype_tile(
        self,
        well_id: str,
        tile_id: str
    ) -> Dict[str, Any]:
        """Genotype a single tile.
        
        This method should be implemented by subclasses to perform the actual
        genotyping. The base implementation raises NotImplementedError.
        
        Args:
            well_id: Well identifier
            tile_id: Tile identifier
            
        Returns:
            Dictionary containing genotyping results
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Genotyping not implemented")
        
    def run(self, wells: Optional[List[str]] = None) -> None:
        """Run the genotyping pipeline.
        
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
                    # Genotype tile
                    results = self.genotype_tile(well_id, tile_id)
                    
                    # Save results
                    self.save_results(well_id, tile_id, results)
                    
                except Exception as e:
                    self.logger.error(
                        f"Error processing tile {tile_id} in well {well_id}: {e}"
                    )
                    continue
                    
        self.logger.info("Genotyping completed successfully") 