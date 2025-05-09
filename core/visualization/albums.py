"""Module for generating cell image albums with phenotype information."""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage import exposure, transform, draw

from ImageAnalysis.core.pipeline import Pipeline
from ImageAnalysis.utils.io import ImageLoader

class AlbumGenerationPipeline(Pipeline):
    """Pipeline for generating cell image albums.
    
    Creates organized grids of cell images with optional phenotype
    and genotype information overlaid.
    """
    
    def __init__(
        self,
        input_file: Union[str, Path],
        phenotyping_dir: Union[str, Path],
        channels: List[str],
        output_dir: Optional[Union[str, Path]] = None,
        config_file: Optional[Union[str, Path]] = None,
        window_size: int = 100,
        grid_size: Tuple[int, int] = (5, 5),
        channel_colors: Optional[Dict[str, str]] = None,
        contrast_limits: Optional[Dict[str, Tuple[float, float]]] = None,
        dpi: int = 300
    ):
        """Initialize the album generation pipeline.
        
        Args:
            input_file: Path to input ND2 file
            phenotyping_dir: Directory containing phenotyping results
            channels: List of channel names to include
            output_dir: Optional output directory
            config_file: Optional path to JSON configuration file
            window_size: Size of cell window in pixels
            grid_size: Tuple of (rows, cols) for image grid
            channel_colors: Optional dict mapping channels to colors
            contrast_limits: Optional dict of (min, max) per channel
            dpi: DPI for saved images
        """
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = os.path.join(phenotyping_dir, 'albums')
            
        super().__init__(
            input_file=input_file,
            output_dir=output_dir,
            config_file=config_file
        )
        
        # Store parameters
        self.phenotyping_dir = Path(phenotyping_dir)
        self.channels = channels
        self.window_size = window_size
        self.grid_size = grid_size
        self.dpi = dpi
        
        # Set default channel colors if not provided
        if channel_colors is None:
            default_colors = ['blue', 'green', 'red', 'magenta', 'yellow', 'cyan']
            self.channel_colors = {
                channel: color for channel, color in 
                zip(channels, default_colors[:len(channels)])
            }
        else:
            self.channel_colors = channel_colors
            
        self.contrast_limits = contrast_limits or {}
        
        # Initialize image loader
        self.image_loader = ImageLoader(self.input_file)
        
    def validate_inputs(self) -> None:
        """Validate pipeline inputs.
        
        Raises:
            ValueError: If inputs are invalid
        """
        super().validate_inputs()
        
        # Check phenotyping directory exists
        if not self.phenotyping_dir.exists():
            raise ValueError(f"Phenotyping directory not found: {self.phenotyping_dir}")
            
        # Validate channels
        if not self.channels:
            raise ValueError("At least one channel must be specified")
            
        # Validate grid size
        if any(x <= 0 for x in self.grid_size):
            raise ValueError("Grid dimensions must be positive")
            
        # Validate window size
        if self.window_size <= 0:
            raise ValueError("Window size must be positive")
            
    def crop_cell(
        self,
        image: np.ndarray,
        center: Tuple[int, int],
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Crop a window around a cell.
        
        Args:
            image: Input image
            center: (y, x) coordinates of cell center
            mask: Optional cell mask for precise cropping
            
        Returns:
            Cropped image window
        """
        # Calculate crop coordinates
        y, x = center
        half_size = self.window_size // 2
        
        y_start = max(0, y - half_size)
        y_end = min(image.shape[0], y + half_size)
        x_start = max(0, x - half_size)
        x_end = min(image.shape[1], x + half_size)
        
        # Crop image
        cropped = image[y_start:y_end, x_start:x_end]
        
        # Pad if necessary
        if cropped.shape != (self.window_size, self.window_size):
            padded = np.zeros((self.window_size, self.window_size))
            y_offset = half_size - y
            x_offset = half_size - x
            
            y_slice = slice(max(0, y_offset), 
                          min(self.window_size, y_offset + cropped.shape[0]))
            x_slice = slice(max(0, x_offset),
                          min(self.window_size, x_offset + cropped.shape[1]))
            
            padded[y_slice, x_slice] = cropped
            cropped = padded
            
        return cropped
        
    def enhance_contrast(
        self,
        image: np.ndarray,
        channel: str
    ) -> np.ndarray:
        """Enhance image contrast for visualization.
        
        Args:
            image: Input image
            channel: Channel name for contrast limits
            
        Returns:
            Contrast-enhanced image
        """
        # Get contrast limits for this channel
        if channel in self.contrast_limits:
            vmin, vmax = self.contrast_limits[channel]
            enhanced = exposure.rescale_intensity(
                image, in_range=(vmin, vmax)
            )
        else:
            # Auto-contrast using percentile limits
            p2, p98 = np.percentile(image[image > 0], (2, 98))
            enhanced = exposure.rescale_intensity(
                image, in_range=(p2, p98)
            )
            
        return enhanced
        
    def create_composite(
        self,
        channels: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Create RGB composite from multiple channels.
        
        Args:
            channels: Dictionary of channel images
            
        Returns:
            RGB composite image
        """
        # Initialize RGB image
        composite = np.zeros((*channels[list(channels.keys())[0]].shape, 3))
        
        # Add each channel
        for channel_name, image in channels.items():
            # Get color for this channel
            color = self.channel_colors[channel_name]
            
            # Convert color name to RGB
            rgb = np.array(plt.cm.colors.to_rgb(color))
            
            # Add to composite
            for i in range(3):
                composite[..., i] += image * rgb[i]
                
        # Normalize to [0, 1]
        composite = np.clip(composite, 0, 1)
        
        return composite
        
    def add_scale_bar(
        self,
        ax: plt.Axes,
        pixel_size: float,
        bar_length_um: float = 10.0,
        location: str = 'lower right',
        color: str = 'white',
        height_fraction: float = 0.02,
        padding_fraction: float = 0.1
    ) -> None:
        """Add scale bar to image.
        
        Args:
            ax: Matplotlib axes
            pixel_size: Size of each pixel in microns
            bar_length_um: Length of scale bar in microns
            location: Location of scale bar
            color: Color of scale bar
            height_fraction: Height of bar relative to image
            padding_fraction: Padding from image edge
        """
        # Calculate bar dimensions in pixels
        bar_length_px = bar_length_um / pixel_size
        fig_height_px = ax.get_window_extent().height
        bar_height_px = height_fraction * fig_height_px
        
        # Calculate position
        padding_px = padding_fraction * fig_height_px
        if 'lower' in location:
            y = padding_px
        else:
            y = fig_height_px - padding_px - bar_height_px
            
        if 'right' in location:
            x = fig_height_px - padding_px - bar_length_px
        else:
            x = padding_px
            
        # Add bar
        rect = plt.Rectangle(
            (x, y),
            bar_length_px,
            bar_height_px,
            color=color,
            transform=ax.transData
        )
        ax.add_patch(rect)
        
        # Add label
        ax.text(
            x + bar_length_px/2,
            y - padding_px,
            f'{bar_length_um:.0f} μm',
            color=color,
            ha='center',
            va='top'
        )
        
    def create_cell_grid(
        self,
        cells: List[Dict[str, Any]],
        well_id: str,
        tile_id: str,
        pixel_size: float = 1.0,
        sort_by: Optional[str] = None,
        descending: bool = True
    ) -> None:
        """Create and save grid of cell images.
        
        Args:
            cells: List of cell data dictionaries
            well_id: Well identifier
            tile_id: Tile identifier
            pixel_size: Physical size of each pixel in microns
            sort_by: Optional metric to sort cells by
            descending: Sort order if sort_by is specified
        """
        # Sort cells if requested
        if sort_by is not None:
            cells = sorted(
                cells,
                key=lambda x: x['metrics'].get(sort_by, 0),
                reverse=descending
            )
            
        # Calculate grid dimensions
        n_rows, n_cols = self.grid_size
        n_cells = min(len(cells), n_rows * n_cols)
        
        # Create figure
        fig = plt.figure(
            figsize=(2*n_cols, 2*n_rows),
            dpi=self.dpi
        )
        gs = GridSpec(n_rows, n_cols, figure=fig)
        
        # Load image data
        image_data = self.image_loader.load_tile(well_id, tile_id)
        
        # Process each cell
        for i, cell in enumerate(cells[:n_cells]):
            # Get cell position and metrics
            row = i // n_cols
            col = i % n_cols
            
            # Get cell center
            center_y = cell['metrics']['location']['centroid_y']
            center_x = cell['metrics']['location']['centroid_x']
            
            # Create subplot
            ax = fig.add_subplot(gs[row, col])
            
            # Process each channel
            channel_images = {}
            for j, channel in enumerate(self.channels):
                # Crop channel image
                channel_img = self.crop_cell(
                    image_data[..., j],
                    (int(center_y), int(center_x))
                )
                
                # Enhance contrast
                channel_img = self.enhance_contrast(channel_img, channel)
                
                channel_images[channel] = channel_img
                
            # Create composite
            composite = self.create_composite(channel_images)
            
            # Display image
            ax.imshow(composite)
            
            # Add scale bar
            self.add_scale_bar(ax, pixel_size)
            
            # Add metrics if available
            if 'metrics' in cell:
                metric_text = []
                if 'cell_area' in cell['metrics']:
                    area = cell['metrics']['cell_area']['area']
                    metric_text.append(f'Area: {area:.0f} μm²')
                    
                if 'intensity' in cell['metrics']:
                    for channel, data in cell['metrics']['intensity'].items():
                        mean_int = data['cell']['mean_intensity']
                        metric_text.append(f'{channel}: {mean_int:.1f}')
                        
                if metric_text:
                    ax.text(
                        0.02, 0.98,
                        '\n'.join(metric_text),
                        transform=ax.transAxes,
                        color='white',
                        fontsize=8,
                        va='top'
                    )
                    
            # Add genotype if available
            if 'genotype' in cell:
                ax.text(
                    0.02, 0.02,
                    f"Barcode: {cell['genotype']['barcode']}",
                    transform=ax.transAxes,
                    color='white',
                    fontsize=8,
                    va='bottom'
                )
                
            # Remove axes
            ax.axis('off')
            
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(
            self.output_dir,
            well_id,
            f'{tile_id}_grid.png'
        )
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
    def process_tile(
        self,
        well_id: str,
        tile_id: str
    ) -> None:
        """Process a single tile.
        
        Args:
            well_id: Well identifier
            tile_id: Tile identifier
        """
        # Load phenotyping results
        results_file = os.path.join(
            self.phenotyping_dir,
            well_id,
            f'{tile_id}.json'
        )
        
        if not os.path.exists(results_file):
            self.logger.warning(f"No phenotyping results found for {well_id}/{tile_id}")
            return
            
        with open(results_file, 'r') as f:
            phenotype_data = json.load(f)
            
        # Create cell grid
        self.create_cell_grid(
            cells=phenotype_data['cells'],
            well_id=well_id,
            tile_id=tile_id
        )
        
        self.logger.info(f"Created cell grid for {well_id}/{tile_id}")
        
    def run(self, wells: Optional[List[str]] = None) -> None:
        """Run the album generation pipeline.
        
        Args:
            wells: Optional list of wells to process (defaults to all wells)
        """
        # Validate inputs
        self.validate_inputs()
        
        # Get wells to process
        if wells is None:
            wells = [d.name for d in self.phenotyping_dir.iterdir()
                    if d.is_dir()]
            
        # Process each well
        for well_id in wells:
            self.logger.info(f"Processing well: {well_id}")
            
            # Get tiles for well
            tiles = [f.stem for f in (self.phenotyping_dir / well_id).glob('*.json')
                    if f.name != 'summary.json']
            
            # Process each tile
            for tile_id in tiles:
                try:
                    self.process_tile(well_id, tile_id)
                except Exception as e:
                    self.logger.error(
                        f"Error processing tile {tile_id} in well {well_id}: {e}"
                    )
                    continue
                    
        self.logger.info("Album generation completed successfully") 