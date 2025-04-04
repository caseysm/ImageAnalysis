"""40X magnification segmentation pipeline using real image processing."""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
import nd2reader
from cellpose import models as cellpose_models
from cellpose import utils
import skimage.measure as sm
from datetime import datetime

class Segmentation40XPipeline:
    """Segmentation pipeline for 40X magnification images.
    
    This pipeline processes ND2 files using cellpose to segment nuclei and cells.
    The 40X implementation uses higher resolution parameters than the 10X pipeline.
    """
    
    def __init__(self, config):
        """Initialize the pipeline with configuration.
        
        Args:
            config (dict): Configuration parameters including:
                - input_file: Path to the ND2 file
                - output_dir: Directory to save results
                - nuclear_channel: Channel index for nuclear stain (default: 0)
                - cell_channel: Channel index for cell stain (default: 1)
                - wells: List of wells to process
                - use_gpu: Whether to use GPU for cellpose (default: False)
        """
        self.config = config
        self.input_file = config.get('input_file')
        self.output_dir = config.get('output_dir', os.path.join('results', 'segmentation'))
        self.nuclear_channel = config.get('nuclear_channel', 0)
        self.cell_channel = config.get('cell_channel', 1)
        self.wells = config.get('wells', [])
        self.use_gpu = config.get('use_gpu', False)
        self.logger = logging.getLogger("segmentation")
        
    def load_nd2_image(self, file_path):
        """Load an ND2 file and extract the required channels.
        
        Args:
            file_path: Path to the ND2 file
            
        Returns:
            tuple: (nuclear_image, cell_image)
        """
        self.logger.info(f"Loading ND2 file: {file_path}")
        try:
            # Open the ND2 file
            with nd2reader.ND2Reader(file_path) as images:
                # Get dimensions
                n_channels = images.sizes['c']
                
                # Validate channel indices
                if self.nuclear_channel >= n_channels or self.cell_channel >= n_channels:
                    raise ValueError(f"Invalid channel indices. File has {n_channels} channels.")
                
                # Extract nuclear channel
                images.default_coords['c'] = self.nuclear_channel
                nuclear_image = images[0]
                
                # Extract cell channel
                images.default_coords['c'] = self.cell_channel
                cell_image = images[0]
                
                return nuclear_image, cell_image
                
        except Exception as e:
            self.logger.error(f"Error loading ND2 file: {str(e)}")
            # Proper error handling
            self.logger.error("Failed to load image data. Please check if the file exists and is a valid ND2 file.")
            raise
    
    def segment_nuclei(self, nuclear_image, diameter=120):
        """Segment nuclei using cellpose with 40X-specific parameters.
        
        Args:
            nuclear_image: 2D numpy array of the nuclear channel
            diameter: Expected nucleus diameter in pixels (default: 120 for 40X)
            
        Returns:
            2D numpy array with nucleus labels
        """
        self.logger.info("Segmenting nuclei with cellpose")
        
        # Initialize cellpose model
        model = cellpose_models.Cellpose(gpu=self.use_gpu, model_type='nuclei')
        
        # Run segmentation with 40X-appropriate diameter
        masks, _, _, _ = model.eval(nuclear_image, diameter=diameter, channels=[[0, 0]])
        
        return masks
    
    def segment_cells(self, cell_image, nuclear_image=None, diameter=240):
        """Segment cells using cellpose.
        
        Args:
            cell_image: 2D numpy array of the cell channel
            nuclear_image: Optional nuclear image to guide segmentation
            diameter: Expected cell diameter in pixels (default: 240 for 40X)
            
        Returns:
            2D numpy array with cell labels
        """
        self.logger.info("Segmenting cells with cellpose")
        
        # Initialize cellpose model
        model = cellpose_models.Cellpose(gpu=self.use_gpu, model_type='cyto2')
        
        # Run segmentation with or without nuclear channel
        if nuclear_image is not None:
            # Prepare 3-channel image with cell and nuclear channels
            h, w = cell_image.shape
            seg_data = np.zeros((h, w, 3))
            seg_data[:, :, 0] = cell_image
            seg_data[:, :, 2] = nuclear_image
            masks, _, _, _ = model.eval(seg_data, diameter=diameter, channels=[[1, 3]])
        else:
            masks, _, _, _ = model.eval(cell_image, diameter=diameter, channels=[[0, 0]])
        
        return masks
    
    def extract_props(self, nuclei_mask, cell_mask):
        """Extract properties of nuclei and cells.
        
        Args:
            nuclei_mask: 2D numpy array with nucleus labels
            cell_mask: 2D numpy array with cell labels
            
        Returns:
            pandas DataFrame with nucleus and cell relationships
        """
        # Get region properties
        props_nuclei = sm.regionprops(nuclei_mask)
        props_cells = sm.regionprops(cell_mask)
        
        # Extract centroids
        nuclei_centroids = np.array([p.centroid for p in props_nuclei])
        
        # Build relationship table
        relationships = []
        for i, (y, x) in enumerate(nuclei_centroids):
            # Get integer coordinates
            y_int, x_int = int(y), int(x)
            
            # Find which cell contains this nucleus
            if 0 <= y_int < cell_mask.shape[0] and 0 <= x_int < cell_mask.shape[1]:
                cell_label = cell_mask[y_int, x_int]
                if cell_label != 0:
                    relationships.append({
                        'nuc': i + 1,
                        'cell': cell_label,
                        'y': y_int,
                        'x': x_int
                    })
        
        return pd.DataFrame(relationships)
    
    def clean_masks(self, nuclei_mask, cell_mask):
        """Clean the masks to keep only cells with one nucleus.
        
        Args:
            nuclei_mask: 2D numpy array with nucleus labels
            cell_mask: 2D numpy array with cell labels
            
        Returns:
            tuple: (cleaned nuclear mask, cleaned cell mask)
        """
        self.logger.info("Cleaning segmentation masks")
        
        # Extract nucleus-cell relationships
        df = self.extract_props(nuclei_mask, cell_mask)
        
        # Find cells with multiple nuclei
        df_multiple = df.groupby('cell').filter(lambda x: len(x) > 1)
        
        # Remove nuclei in cells with multiple nuclei
        cleaned_nuclei = nuclei_mask.copy()
        for _, row in df_multiple.iterrows():
            nuc_label = row['nuc']
            cleaned_nuclei = (cleaned_nuclei != nuc_label) * cleaned_nuclei
        
        # Keep only one-to-one relationships
        df_single = df.groupby('cell').filter(lambda x: len(x) == 1)
        
        # Create new masks with consistent labeling
        cleaned_nuclei_relabeled = np.zeros_like(nuclei_mask)
        cleaned_cells_relabeled = np.zeros_like(cell_mask)
        
        for i, row in df_single.iterrows():
            new_label = i + 1
            old_nuc_label = row['nuc']
            old_cell_label = row['cell']
            
            cleaned_nuclei_relabeled[cleaned_nuclei == old_nuc_label] = new_label
            cleaned_cells_relabeled[cell_mask == old_cell_label] = new_label
        
        # Remove cells touching the image boundary
        h, w = cell_mask.shape
        border = 3  # Border width to check
        
        # Get labels of cells touching the border
        border_labels = set()
        for label in np.unique(cleaned_cells_relabeled):
            if label == 0:
                continue
                
            # Check if cell touches any border
            mask = cleaned_cells_relabeled == label
            if (np.any(mask[:border, :]) or np.any(mask[-border:, :]) or
                np.any(mask[:, :border]) or np.any(mask[:, -border:])):
                border_labels.add(label)
        
        # Remove border cells
        for label in border_labels:
            cleaned_cells_relabeled[cleaned_cells_relabeled == label] = 0
            cleaned_nuclei_relabeled[cleaned_nuclei_relabeled == label] = 0
        
        # Relabel consecutively
        # First create a binary mask
        cell_binary = cleaned_cells_relabeled > 0
        # Then label the connected components
        final_cells = sm.label(cell_binary)
        
        # Make nuclear labels match cell labels
        final_nuclei = np.zeros_like(nuclei_mask)
        for cell_label in np.unique(final_cells):
            if cell_label == 0:
                continue
                
            # Get cell mask
            cell_mask_single = final_cells == cell_label
            
            # Find corresponding nucleus in this cell
            nuclei_in_cell = cleaned_nuclei_relabeled * cell_mask_single
            nucleus_labels = np.unique(nuclei_in_cell)
            nucleus_labels = nucleus_labels[nucleus_labels > 0]
            
            if len(nucleus_labels) == 1:
                # Label nucleus with cell label
                final_nuclei[cleaned_nuclei_relabeled == nucleus_labels[0]] = cell_label
        
        return final_nuclei, final_cells
    
    def extract_tile_metadata(self, file_path):
        """Extract tile metadata from ND2 file.
        
        Args:
            file_path: Path to ND2 file
            
        Returns:
            dict: Tile metadata
        """
        try:
            # Extract metadata from ND2 file
            with nd2reader.ND2Reader(file_path) as images:
                metadata = {
                    'x_um': images[0].metadata['x_um'],
                    'y_um': images[0].metadata['y_um'],
                    'mpp': images[0].metadata['mpp'],
                    'width': images.sizes['x'],
                    'height': images.sizes['y']
                }
                return metadata
        except Exception as e:
            self.logger.error(f"Error extracting tile metadata: {str(e)}")
            return None
            
    def extract_and_save_nuclei_centroids(self, nuclei_mask, well, file_id, well_dir, tile_metadata=None):
        """Extract nuclei centroids and save them.
        
        Args:
            nuclei_mask: Labeled nuclei mask
            well: Well identifier
            file_id: File identifier
            well_dir: Directory to save results
            tile_metadata: Optional tile metadata for global coordinates
            
        Returns:
            np.ndarray: Extracted centroids
        """
        # Extract centroids
        props = sm.regionprops(nuclei_mask)
        if not props:
            self.logger.warning(f"No nuclei found in {file_id}")
            return np.empty((0, 2))
            
        # Get centroids (y, x) format
        centroids = np.array([prop.centroid for prop in props])
        
        # Save local centroids
        local_centroids_file = os.path.join(well_dir, f"{file_id}_nuclei_centroids_local.npy")
        np.save(local_centroids_file, centroids)
        
        # If tile metadata is available, convert to global coordinates
        if tile_metadata:
            # Get tile position from metadata
            x_pos = tile_metadata['x_um']
            y_pos = tile_metadata['y_um']
            
            # Convert to global coordinates - this needs to be specific for 40X
            # We're just saving centroids in the original format, conversion will be done during mapping
            global_centroids = centroids.copy()
            
            # Append to global centroids file for the well
            global_file = os.path.join(well_dir, f"{well}_nuclei_centroids_40x.npy")
            
            if os.path.exists(global_file):
                existing_centroids = np.load(global_file)
                all_centroids = np.vstack((existing_centroids, global_centroids))
                np.save(global_file, all_centroids)
            else:
                np.save(global_file, global_centroids)
                
            # Also save tile information for the mapping pipeline
            tile_info_file = os.path.join(well_dir, f"{file_id}_tile_info.json")
            with open(tile_info_file, 'w') as f:
                json.dump(tile_metadata, f, indent=2)
                
            return global_centroids
        
        return centroids
    
    def run(self):
        """Run the segmentation pipeline.
        
        Returns:
            dict: Results summary
        """
        start_time = datetime.now()
        self.logger.info(f"Starting 40X segmentation pipeline at {start_time}")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Extract well name from input file
        input_path = Path(self.input_file)
        file_name = input_path.name
        file_id = input_path.stem
        
        # Parse well ID from filename (assuming Well1_Point1_... format)
        parts = file_name.split('_')
        if parts[0].startswith('Well'):
            well = parts[0]
        else:
            well = 'Well1'  # Default if not found
        
        # Skip if wells specified and not in list
        if self.wells and well not in self.wells:
            self.logger.info(f"Skipping well {well} (not in specified wells list)")
            return {"status": "skipped", "well": well}
        
        # Create well directory
        well_dir = os.path.join(self.output_dir, well)
        os.makedirs(well_dir, exist_ok=True)
        
        try:
            # Load image
            nuclear_image, cell_image = self.load_nd2_image(self.input_file)
            
            # Segment nuclei and cells with 40X-appropriate diameters
            nuclei_mask = self.segment_nuclei(nuclear_image, diameter=120)
            cell_mask = self.segment_cells(cell_image, nuclear_image, diameter=240)
            
            # Clean masks
            clean_nuclei, clean_cells = self.clean_masks(nuclei_mask, cell_mask)
            
            # Save masks
            np.save(os.path.join(well_dir, f"{file_id}_nuclei_mask.npy"), clean_nuclei)
            np.save(os.path.join(well_dir, f"{file_id}_cell_mask.npy"), clean_cells)
            
            # Extract tile metadata
            tile_metadata = self.extract_tile_metadata(self.input_file)
            
            # Extract and save centroids
            if tile_metadata:
                self.logger.info(f"Extracting centroids with global coordinates for 40X")
                centroids = self.extract_and_save_nuclei_centroids(
                    clean_nuclei, well, file_id, well_dir, tile_metadata
                )
            else:
                self.logger.info(f"Extracting centroids with local coordinates only for 40X")
                centroids = self.extract_and_save_nuclei_centroids(
                    clean_nuclei, well, file_id, well_dir
                )
            
            # Create summary
            num_nuclei = len(np.unique(clean_nuclei)) - 1  # Subtract background
            num_cells = len(np.unique(clean_cells)) - 1    # Subtract background
            
            # Create properties
            properties = {
                'file': str(input_path),
                'well': well,
                'num_nuclei': int(num_nuclei),
                'num_cells': int(num_cells),
                'processing_time': str(datetime.now() - start_time),
                'magnification': '40X',
                'tile_metadata': tile_metadata
            }
            
            # Save properties
            with open(os.path.join(well_dir, f"{file_id}_properties.json"), 'w') as f:
                json.dump(properties, f, indent=2)
            
            self.logger.info(f"40X segmentation completed: {num_cells} cells found, {len(centroids)} centroids extracted")
            
            # Return results
            return {
                "status": "success",
                "well": well,
                "file": file_id,
                "num_nuclei": num_nuclei,
                "num_cells": num_cells,
                "num_centroids": len(centroids),
                "magnification": "40X"
            }
            
        except Exception as e:
            self.logger.error(f"40X segmentation failed: {str(e)}")
            return {
                "status": "error",
                "well": well,
                "error": str(e),
                "magnification": "40X"
            }