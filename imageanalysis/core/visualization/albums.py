"""Album generation module for cell visualization based on original pipeline."""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
from scipy import ndimage as ndi
from typing import Dict, List, Tuple, Union, Optional

class AlbumGenerationPipeline:
    """Pipeline for generating cell albums and single cell visualizations.
    
    This pipeline creates visual albums of cells organized by groups (e.g., genotypes)
    and allows for visualization of cellular features and phenotypes.
    """
    
    def __init__(self, config):
        """Initialize the pipeline with configuration.
        
        Args:
            config (dict): Configuration parameters including:
                - phenotyping_dir: Directory with phenotyping results
                - segmentation_dir: Directory with segmentation results
                - output_dir: Directory to save albums
                - wells: List of wells to process
                - channels: List of channel names in the images (e.g. ['DAPI', 'GFP', 'RFP'])
                - grid_size: Tuple of (rows, columns) for the album grid layout (default: (5, 10))
                - cell_size: Size of individual cell thumbnails in pixels (default: 128)
                - group_by: Column to group cells by (default: None, all cells in one album)
                - sort_by: Column to sort cells within a group (default: None)
                - colormap: Colormap for single-channel images (default: 'viridis')
                - intensity_percentile: Percentile for intensity normalization (default: 99.5)
        """
        self.config = config
        self.phenotyping_dir = config.get('phenotyping_dir')
        self.segmentation_dir = config.get('segmentation_dir', None)
        self.output_dir = config.get('output_dir', os.path.join('results', 'albums'))
        self.wells = config.get('wells', [])
        self.channels = config.get('channels', ['DAPI', 'GFP', 'RFP'])
        self.grid_size = config.get('grid_size', (5, 10))
        self.cell_size = config.get('cell_size', 128)
        self.group_by = config.get('group_by', None)
        self.sort_by = config.get('sort_by', None)
        self.colormap = config.get('colormap', 'viridis')
        self.intensity_percentile = config.get('intensity_percentile', 99.5)
        self.logger = logging.getLogger("albums")
        
        # Configure matplotlib to avoid X11 issues
        matplotlib.use('Agg')
        
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range.
        
        Args:
            image: Image array
            
        Returns:
            Normalized image
        """
        if np.all(image == 0):
            return image
            
        min_val = np.min(image)
        max_val = np.percentile(image, self.intensity_percentile)
        
        if max_val > min_val:
            return np.clip((image - min_val) / (max_val - min_val), 0, 1)
        else:
            return np.zeros_like(image)
    
    def load_single_cell_image(self, image_data: np.ndarray, cell_mask: np.ndarray, cell_id: int) -> np.ndarray:
        """Extract and process a single cell from an image.
        
        Args:
            image_data: Multi-channel image data (channels, height, width)
            cell_mask: Cell segmentation mask
            cell_id: ID of the cell to extract
            
        Returns:
            Processed single cell image with dimensions (height, width, channels)
        """
        # Create cell mask for this specific cell
        single_cell_mask = (cell_mask == cell_id)
        
        if not np.any(single_cell_mask):
            self.logger.warning(f"Cell ID {cell_id} not found in mask")
            return None
            
        # Get bounding box
        y_indices, x_indices = np.where(single_cell_mask)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        
        # Add padding to make the crop more generous
        padding = 5
        y_min = max(0, y_min - padding)
        y_max = min(cell_mask.shape[0], y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(cell_mask.shape[1], x_max + padding)
        
        # Crop mask
        cropped_mask = single_cell_mask[y_min:y_max, x_min:x_max]
        
        # Crop and apply mask to each channel
        cell_image = np.zeros((y_max-y_min, x_max-x_min, len(self.channels)))
        
        for i, channel in enumerate(range(min(image_data.shape[0], len(self.channels)))):
            channel_data = image_data[channel, y_min:y_max, x_min:x_max]
            cell_image[..., i] = channel_data * cropped_mask
            
        return cell_image
    
    def resize_cell_image(self, cell_image: np.ndarray, size: int = 128) -> np.ndarray:
        """Resize cell image to a square of specified size.
        
        Args:
            cell_image: Cell image with dimensions (height, width, channels)
            size: Size of the output square image
            
        Returns:
            Resized cell image
        """
        if cell_image is None:
            return np.zeros((size, size, len(self.channels)))
            
        h, w, c = cell_image.shape
        
        # Calculate the largest dimension
        max_dim = max(h, w)
        
        # Create a square image filled with zeros
        square_image = np.zeros((max_dim, max_dim, c))
        
        # Center the cell image in the square
        y_offset = (max_dim - h) // 2
        x_offset = (max_dim - w) // 2
        
        square_image[y_offset:y_offset+h, x_offset:x_offset+w] = cell_image
        
        # Resize to the specified size
        resized_image = np.zeros((size, size, c))
        
        for i in range(c):
            # Use scipy to resize each channel
            resized_image[..., i] = ndi.zoom(
                square_image[..., i], 
                size / max_dim, 
                order=1  # Linear interpolation
            )
            
        return resized_image
    
    def create_single_cell_thumbnails(self, 
                                    image_data: np.ndarray, 
                                    cell_mask: np.ndarray,
                                    phenotypes_df: pd.DataFrame,
                                    output_dir: str) -> Dict[int, str]:
        """Create thumbnails for each cell in the phenotypes DataFrame.
        
        Args:
            image_data: Multi-channel image data (channels, height, width)
            cell_mask: Cell segmentation mask
            phenotypes_df: DataFrame with phenotype data
            output_dir: Directory to save thumbnails
            
        Returns:
            Dictionary mapping cell_ids to thumbnail paths
        """
        self.logger.info(f"Creating {len(phenotypes_df)} single cell thumbnails")
        
        # Create output directory
        cells_dir = os.path.join(output_dir, 'cells')
        os.makedirs(cells_dir, exist_ok=True)
        
        thumbnails = {}
        
        for idx, row in phenotypes_df.iterrows():
            try:
                cell_id = int(row['cell_id'])
                
                # Extract and process cell image
                cell_image = self.load_single_cell_image(image_data, cell_mask, cell_id)
                
                if cell_image is None:
                    continue
                    
                # Normalize each channel
                for i in range(cell_image.shape[2]):
                    cell_image[..., i] = self.normalize_image(cell_image[..., i])
                    
                # Resize to standard size
                resized_cell = self.resize_cell_image(cell_image, self.cell_size)
                
                # Save thumbnail
                thumbnail_path = os.path.join(cells_dir, f"cell_{cell_id}.npy")
                np.save(thumbnail_path, resized_cell)
                
                thumbnails[cell_id] = thumbnail_path
                
            except Exception as e:
                self.logger.error(f"Error processing cell {cell_id}: {str(e)}")
                
        return thumbnails
    
    def create_album_image(self, 
                         cell_thumbnails: Dict[int, str],
                         selected_cells: List[int],
                         grid_size: Tuple[int, int]) -> np.ndarray:
        """Create an album image by arranging cell thumbnails in a grid.
        
        Args:
            cell_thumbnails: Dictionary mapping cell_ids to thumbnail paths
            selected_cells: List of cell IDs to include in the album
            grid_size: Tuple of (rows, columns) for the grid layout
            
        Returns:
            Album image as numpy array
        """
        rows, cols = grid_size
        
        # Limit selected cells to the grid size
        selected_cells = selected_cells[:rows*cols]
        
        # Load the first thumbnail to get dimensions
        first_path = list(cell_thumbnails.values())[0]
        first_thumb = np.load(first_path)
        cell_size = first_thumb.shape[0]
        channels = first_thumb.shape[2]
        
        # Create empty album
        album = np.zeros((rows * cell_size, cols * cell_size, channels))
        
        # Add cells to album
        for i, cell_id in enumerate(selected_cells):
            if cell_id in cell_thumbnails:
                # Calculate position in grid
                row = i // cols
                col = i % cols
                
                # Load thumbnail
                try:
                    thumbnail = np.load(cell_thumbnails[cell_id])
                    # Add to album
                    album[row*cell_size:(row+1)*cell_size, 
                         col*cell_size:(col+1)*cell_size] = thumbnail
                except Exception as e:
                    self.logger.error(f"Error loading thumbnail for cell {cell_id}: {str(e)}")
                
        return album
    
    def select_cells(self, 
                   phenotypes_df: pd.DataFrame, 
                   group: Optional[str] = None, 
                   max_cells: int = None) -> List[int]:
        """Select cells for an album, optionally filtering by group.
        
        Args:
            phenotypes_df: DataFrame with phenotype data
            group: Value to filter by in group_by column
            max_cells: Maximum number of cells to select
            
        Returns:
            List of selected cell IDs
        """
        # Filter by group if specified
        if group is not None and self.group_by is not None:
            df = phenotypes_df[phenotypes_df[self.group_by] == group]
        else:
            df = phenotypes_df
            
        # Sort if specified
        if self.sort_by is not None and self.sort_by in df.columns:
            df = df.sort_values(by=self.sort_by, ascending=False)
            
        # Limit to max_cells
        if max_cells is not None:
            df = df.head(max_cells)
            
        # Return cell IDs
        return df['cell_id'].tolist()
    
    def process_well(self, well: str) -> Dict:
        """Process a single well.
        
        Args:
            well: Well identifier
            
        Returns:
            Dictionary with results
        """
        self.logger.info(f"Processing well {well}")
        
        try:
            # Create output directory
            well_out_dir = os.path.join(self.output_dir, well)
            os.makedirs(well_out_dir, exist_ok=True)
            
            # Load phenotype data
            pheno_file = os.path.join(self.phenotyping_dir, well, f"{well}_phenotypes.csv")
            if not os.path.exists(pheno_file):
                return {"status": "error", "message": f"No phenotype file found for well {well}"}
                
            phenotypes_df = pd.read_csv(pheno_file)
            
            # Load segmentation masks
            seg_dir = os.path.join(self.segmentation_dir, well) if self.segmentation_dir else None
            cell_mask_file = None
            
            if seg_dir and os.path.exists(seg_dir):
                cell_mask_files = [f for f in os.listdir(seg_dir) if f.endswith('_cell_mask.npy')]
                if cell_mask_files:
                    cell_mask_file = os.path.join(seg_dir, cell_mask_files[0])
                    
            if cell_mask_file is None:
                return {"status": "error", "message": f"No cell mask found for well {well}"}
                
            cell_mask = np.load(cell_mask_file)
            
            # Load image data from phenotype file
            image_file = os.path.join(self.phenotyping_dir, well, f"{well}_image.npy")
            if not os.path.exists(image_file):
                # If no image file exists, we'll try to create thumbnails from bounding box info in phenotypes
                if all(col in phenotypes_df.columns for col in ['y_min', 'y_max', 'x_min', 'x_max']):
                    self.logger.warning(f"No image file found. Using bounding box info from phenotypes")
                    
                    # Create dummy thumbnails based on phenotype data
                    cell_thumbnails = {}
                    cells_dir = os.path.join(well_out_dir, 'cells')
                    os.makedirs(cells_dir, exist_ok=True)
                    
                    for idx, row in phenotypes_df.iterrows():
                        cell_id = int(row['cell_id'])
                        
                        # Create dummy thumbnail based on phenotype features
                        thumbnail = np.zeros((self.cell_size, self.cell_size, len(self.channels)))
                        
                        # Use phenotype data to create a visual representation
                        for i, channel in enumerate(self.channels[:len(self.channels)]):
                            if f'{channel}_mean_intensity' in row:
                                intensity = float(row[f'{channel}_mean_intensity'])
                                # Create a circle with intensity proportional to the mean intensity
                                y, x = np.ogrid[-self.cell_size//2:self.cell_size//2, -self.cell_size//2:self.cell_size//2]
                                mask = x**2 + y**2 <= (self.cell_size//3)**2
                                thumbnail[mask + self.cell_size//2, mask + self.cell_size//2, i] = intensity
                                
                        # Save thumbnail
                        thumbnail_path = os.path.join(cells_dir, f"cell_{cell_id}.npy")
                        np.save(thumbnail_path, thumbnail)
                        
                        cell_thumbnails[cell_id] = thumbnail_path
                        
                else:
                    return {"status": "error", "message": f"No image data found for well {well}"}
            else:
                # Load image data
                image_data = np.load(image_file)
                
                # Create thumbnails
                cell_thumbnails = self.create_single_cell_thumbnails(
                    image_data, 
                    cell_mask, 
                    phenotypes_df, 
                    well_out_dir
                )
                
            # Determine how to group cells for albums
            if self.group_by is not None and self.group_by in phenotypes_df.columns:
                groups = phenotypes_df[self.group_by].unique()
                
                albums = {}
                for group in groups:
                    # Select cells for this group
                    selected_cells = self.select_cells(
                        phenotypes_df, 
                        group, 
                        self.grid_size[0] * self.grid_size[1]
                    )
                    
                    if not selected_cells:
                        continue
                        
                    # Create album
                    album = self.create_album_image(
                        cell_thumbnails,
                        selected_cells,
                        self.grid_size
                    )
                    
                    # Save album
                    album_name = f"{well}_{self.group_by}_{group}_album"
                    album_path = os.path.join(well_out_dir, f"{album_name}.npy")
                    np.save(album_path, album)
                    
                    # Save metadata
                    metadata = {
                        'well': well,
                        'group': group,
                        'group_by': self.group_by,
                        'grid_size': self.grid_size,
                        'cells_in_album': len(selected_cells),
                        'cell_ids': selected_cells,
                        'channels': self.channels
                    }
                    
                    metadata_path = os.path.join(well_out_dir, f"{album_name}_metadata.json")
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                        
                    albums[str(group)] = {
                        'album_path': album_path,
                        'metadata_path': metadata_path,
                        'cells_in_album': len(selected_cells)
                    }
                    
                # Save summary
                return {
                    "status": "success",
                    "well": well,
                    "albums": albums,
                    "total_cells": len(phenotypes_df),
                    "cell_thumbnails": len(cell_thumbnails)
                }
                
            else:
                # No grouping, create a single album for all cells
                selected_cells = self.select_cells(
                    phenotypes_df, 
                    None, 
                    self.grid_size[0] * self.grid_size[1]
                )
                
                # Create album
                album = self.create_album_image(
                    cell_thumbnails,
                    selected_cells,
                    self.grid_size
                )
                
                # Save album
                album_path = os.path.join(well_out_dir, f"{well}_album.npy")
                np.save(album_path, album)
                
                # Save metadata
                metadata = {
                    'well': well,
                    'grid_size': self.grid_size,
                    'cells_in_album': len(selected_cells),
                    'cell_ids': selected_cells,
                    'channels': self.channels
                }
                
                metadata_path = os.path.join(well_out_dir, "metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
                return {
                    "status": "success",
                    "well": well,
                    "album_path": album_path,
                    "metadata_path": metadata_path,
                    "cells_in_album": len(selected_cells),
                    "total_cells": len(phenotypes_df),
                    "cell_thumbnails": len(cell_thumbnails)
                }
                
        except Exception as e:
            self.logger.error(f"Error processing well {well}: {str(e)}")
            return {
                "status": "error",
                "well": well,
                "message": str(e)
            }
    
    def save_album_figure(self, album: np.ndarray, output_path: str, title: str = None) -> None:
        """Save album as a figure.
        
        Args:
            album: Album image
            output_path: Output file path
            title: Optional title for the figure
        """
        # Create figure
        dpi = 100
        height, width, _ = album.shape
        figsize = (width / dpi, height / dpi)
        
        fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        
        # Add title if provided
        if title:
            plt.title(title)
            
        # Show album
        if album.shape[2] == 1:
            plt.imshow(album[..., 0], cmap=self.colormap)
        else:
            plt.imshow(album)
            
        plt.axis('off')
        
        # Save figure
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    def run(self) -> Dict:
        """Run the album generation pipeline.
        
        Returns:
            Dictionary with results
        """
        self.logger.info("Starting album generation")
        start_time = datetime.now()
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        results = {}
        
        # Process all wells or only specified wells
        if self.wells:
            well_dirs = [d for d in os.listdir(self.phenotyping_dir) 
                        if os.path.isdir(os.path.join(self.phenotyping_dir, d)) and d in self.wells]
        else:
            well_dirs = [d for d in os.listdir(self.phenotyping_dir) 
                        if os.path.isdir(os.path.join(self.phenotyping_dir, d))]
            
        if not well_dirs:
            return {
                "status": "error",
                "message": "No wells found to process"
            }
            
        # Process each well
        for well in well_dirs:
            results[well] = self.process_well(well)
            
            # Also save as PNG for easy viewing
            if results[well]["status"] == "success" and "album_path" in results[well]:
                album_path = results[well]["album_path"]
                album = np.load(album_path)
                png_path = album_path.replace(".npy", ".png")
                self.save_album_figure(album, png_path, title=well)
                results[well]["album_png"] = png_path
                
        # Create summary
        summary = {
            "status": "success" if any(r["status"] == "success" for r in results.values()) else "error",
            "wells_processed": len(results),
            "wells_successful": sum(1 for r in results.values() if r["status"] == "success"),
            "wells_failed": sum(1 for r in results.values() if r["status"] == "error"),
            "processing_time": str(datetime.now() - start_time)
        }
        
        return {
            **summary,
            "wells": results
        }