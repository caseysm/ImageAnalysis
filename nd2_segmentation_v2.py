#!/usr/bin/env python3
"""
Segmentation script that works directly with ND2 files using the 'nd2' package.
This is an alternative to the TIFF-based approach, leveraging the nd2 package
which works on Apple Silicon (M3 Pro).
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import nd2  # Using the nd2 package (not nd2reader)
from cellpose import models as cellpose_models
import skimage.measure as sm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nd2-segmentation")

class ND2Segmentation:
    """
    Segmentation pipeline for ND2 images.
    Works with either 10X or 40X magnification.
    """
    
    def __init__(self, config):
        """Initialize the pipeline with configuration.
        
        Args:
            config (dict): Configuration parameters including:
                - input_file: Path to ND2 file
                - output_dir: Directory to save results
                - nuclear_channel: Index of nuclear channel (default: 0 for DAPI)
                - cell_channel: Index of cell channel (default: 1)
                - magnification: "10X" or "40X"
                - save_visualization: Whether to save visualizations (default: True)
                - use_gpu: Whether to use GPU for cellpose (default: False)
        """
        self.config = config
        self.input_file = config.get('input_file')
        self.output_dir = config.get('output_dir')
        self.nuclear_channel = config.get('nuclear_channel', 0)
        self.cell_channel = config.get('cell_channel', 1)
        self.magnification = config.get('magnification', '10X')
        self.enable_visualization = config.get('save_visualization', True)
        self.use_gpu = config.get('use_gpu', False)
        
        # Set parameters based on magnification
        if self.magnification == '10X':
            self.nuclei_diameter = 30
            self.cell_diameter = 60
        else:  # 40X
            self.nuclei_diameter = 60
            self.cell_diameter = 120
    
    def load_nd2_images(self):
        """Load images from ND2 file for nuclear and cell channels.
        
        Returns:
            tuple: (nuclear_image, cell_image)
        """
        logger.info(f"Loading ND2 file: {self.input_file}")
        
        try:
            # Load ND2 file
            images = nd2.imread(self.input_file)
            logger.info(f"Loaded ND2 data with shape {images.shape}")
            
            # Extract dimensions and determine how to access channels
            ndim = len(images.shape)
            
            if ndim == 3:  # (c, y, x) format
                # Get the nuclear and cell channels directly
                nuclear_image = images[self.nuclear_channel]
                cell_image = images[self.cell_channel]
            elif ndim == 4:  # Could be (t, c, y, x) or (z, c, y, x)
                # Get the first time point/z-slice
                nuclear_image = images[0, self.nuclear_channel]
                cell_image = images[0, self.cell_channel]
            elif ndim == 5:  # (t, z, c, y, x)
                # Get the first time point and z-slice
                nuclear_image = images[0, 0, self.nuclear_channel]
                cell_image = images[0, 0, self.cell_channel]
            else:
                raise ValueError(f"Unexpected image shape: {images.shape}")
            
            logger.info(f"Extracted nuclear channel with shape {nuclear_image.shape}")
            logger.info(f"Extracted cell channel with shape {cell_image.shape}")
            
            # Normalize images to range 0-255 (as cellpose expects)
            nuclear_image = self._normalize_image(nuclear_image)
            cell_image = self._normalize_image(cell_image)
            
            return nuclear_image, cell_image
                
        except Exception as e:
            logger.error(f"Error loading ND2 file: {e}")
            raise
    
    def _normalize_image(self, image):
        """Normalize image to range 0-255 for cellpose.
        
        Args:
            image: Input image array
            
        Returns:
            np.ndarray: Normalized image
        """
        # Convert to float for normalization
        image = image.astype(np.float32)
        
        # Apply contrast limits (1st and 99th percentiles)
        p1, p99 = np.percentile(image, (1, 99))
        image = np.clip(image, p1, p99)
        
        # Scale to range 0-255
        image = 255 * (image - p1) / (p99 - p1)
        
        # Convert back to uint8
        return image.astype(np.uint8)
    
    def segment_nuclei(self, nuclear_image, diameter=None):
        """Segment nuclei using cellpose.
        
        Args:
            nuclear_image: 2D numpy array of the nuclear channel
            diameter: Expected nucleus diameter in pixels (overrides default)
            
        Returns:
            2D numpy array with nucleus labels
        """
        diameter = diameter or self.nuclei_diameter
        logger.info(f"Segmenting nuclei with cellpose (diameter={diameter})")
        
        # Initialize cellpose model
        model = cellpose_models.CellposeModel(gpu=self.use_gpu, model_type='nuclei')
        
        # Run segmentation - handle both cellpose 3.x (4 returns) and 4.x (3 returns) APIs
        result = model.eval(nuclear_image, diameter=diameter, channels=[0, 0])
        if len(result) == 4:
            masks, _, _, _ = result
        elif len(result) == 3:
            masks, _, _ = result
        else:
            masks = result[0]
        
        logger.info(f"Found {len(np.unique(masks))-1} nuclei")
        return masks
    
    def segment_cells(self, cell_image, nuclear_image=None, diameter=None):
        """Segment cells using cellpose.
        
        Args:
            cell_image: 2D numpy array of the cell channel
            nuclear_image: Optional nuclear image to guide segmentation
            diameter: Expected cell diameter in pixels (overrides default)
            
        Returns:
            2D numpy array with cell labels
        """
        diameter = diameter or self.cell_diameter
        logger.info(f"Segmenting cells with cellpose (diameter={diameter})")
        
        # Initialize cellpose model
        model = cellpose_models.CellposeModel(gpu=self.use_gpu, model_type='cyto2')
        
        # Run segmentation with or without nuclear channel
        if nuclear_image is not None:
            # Prepare 3-channel image with cell and nuclear channels
            h, w = cell_image.shape
            seg_data = np.zeros((h, w, 3))
            seg_data[:, :, 0] = cell_image
            seg_data[:, :, 2] = nuclear_image
            result = model.eval(seg_data, diameter=diameter, channels=[1, 3])
        else:
            result = model.eval(cell_image, diameter=diameter, channels=[0, 0])
            
        # Handle both cellpose 3.x (4 returns) and 4.x (3 returns) APIs
        if len(result) == 4:
            masks, _, _, _ = result
        elif len(result) == 3:
            masks, _, _ = result
        else:
            masks = result[0]
        
        logger.info(f"Found {len(np.unique(masks))-1} cells")
        return masks
    
    def clean_masks(self, nuclei_mask, cell_mask):
        """Clean the masks to keep only cells with one nucleus.
        
        Args:
            nuclei_mask: 2D numpy array with nucleus labels
            cell_mask: 2D numpy array with cell labels
            
        Returns:
            tuple: (cleaned nuclear mask, cleaned cell mask)
        """
        logger.info("Cleaning segmentation masks")
        
        # Extract properties
        props_nuclei = sm.regionprops(nuclei_mask)
        
        # Check if we have any cells or nuclei
        if len(props_nuclei) == 0 or np.max(cell_mask) == 0:
            logger.warning("No cells or nuclei found for cleaning")
            return np.zeros_like(nuclei_mask), np.zeros_like(cell_mask)
        
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
        
        # If no relationships were found, return empty masks
        if not relationships:
            logger.warning("No cell-nucleus relationships found")
            return np.zeros_like(nuclei_mask), np.zeros_like(cell_mask)
        
        df = pd.DataFrame(relationships)
        
        # Find cells with multiple nuclei
        df_multiple = df.groupby('cell').filter(lambda x: len(x) > 1)
        
        # Remove nuclei in cells with multiple nuclei
        cleaned_nuclei = nuclei_mask.copy()
        for _, row in df_multiple.iterrows():
            nuc_label = row['nuc']
            cleaned_nuclei = (cleaned_nuclei != nuc_label) * cleaned_nuclei
        
        # Keep only one-to-one relationships
        df_single = df.groupby('cell').filter(lambda x: len(x) == 1)
        
        # If no single-nucleus cells were found, return empty masks
        if len(df_single) == 0:
            logger.warning("No single-nucleus cells found")
            return np.zeros_like(nuclei_mask), np.zeros_like(cell_mask)
        
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
        border = 3
        
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
        cleaned_cells_relabeled = sm.label(cleaned_cells_relabeled > 0)
        
        # Make nuclear labels match cell labels
        final_nuclei = np.zeros_like(nuclei_mask)
        for cell_label in np.unique(cleaned_cells_relabeled):
            if cell_label == 0:
                continue
                
            # Get cell mask
            cell_mask_single = cleaned_cells_relabeled == cell_label
            
            # Find corresponding nucleus
            nuclei_in_cell = cleaned_nuclei_relabeled * cell_mask_single
            nucleus_labels = np.unique(nuclei_in_cell)
            nucleus_labels = nucleus_labels[nucleus_labels > 0]
            
            if len(nucleus_labels) == 1:
                # Label nucleus with cell label
                final_nuclei[cleaned_nuclei_relabeled == nucleus_labels[0]] = cell_label
        
        logger.info(f"After cleaning: {len(np.unique(final_nuclei))-1} nuclei, {len(np.unique(cleaned_cells_relabeled))-1} cells")
        return final_nuclei, cleaned_cells_relabeled
    
    def save_visualization(self, nuclei_mask, cell_mask, image_name):
        """Save visualization of the segmentation results.
        
        Args:
            nuclei_mask: 2D numpy array with nucleus labels
            cell_mask: 2D numpy array with cell labels
            image_name: Base name for the output files
            
        Returns:
            dict: Paths to visualization files
        """
        if not self.enable_visualization:
            return {}
        
        # Check if we have any cells or nuclei
        if np.max(nuclei_mask) == 0 and np.max(cell_mask) == 0:
            logger.warning("No cells or nuclei to visualize")
            return {}
        
        # Create visualization directory
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create visualizations
        nuclei_outlines = np.zeros((*nuclei_mask.shape, 3), dtype=np.uint8)
        cell_outlines = np.zeros((*cell_mask.shape, 3), dtype=np.uint8)
        
        # Generate random colors for each label
        np.random.seed(42)  # For consistent colors
        
        # Nuclei visualization (red outlines)
        from skimage.segmentation import find_boundaries
        nuclei_boundaries = find_boundaries(nuclei_mask, mode='outer')
        nuclei_outlines[nuclei_boundaries, 0] = 255  # Red channel
        
        # Cell visualization (green outlines)
        cell_boundaries = find_boundaries(cell_mask, mode='outer')
        cell_outlines[cell_boundaries, 1] = 255  # Green channel
        
        # Combined visualization
        combined = nuclei_outlines.copy()
        combined[cell_boundaries, 1] = 255
        
        # Save visualizations
        nuclei_viz_path = os.path.join(viz_dir, f"{image_name}_nuclei.png")
        cell_viz_path = os.path.join(viz_dir, f"{image_name}_cells.png")
        combined_viz_path = os.path.join(viz_dir, f"{image_name}_combined.png")
        
        plt.figure(figsize=(10, 10))
        plt.imshow(nuclei_outlines)
        plt.title(f"Nuclei Segmentation ({len(np.unique(nuclei_mask))-1} nuclei)")
        plt.savefig(nuclei_viz_path, dpi=150)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(cell_outlines)
        plt.title(f"Cell Segmentation ({len(np.unique(cell_mask))-1} cells)")
        plt.savefig(cell_viz_path, dpi=150)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(combined)
        plt.title(f"Combined Segmentation (Nuclei: Red, Cells: Green)")
        plt.savefig(combined_viz_path, dpi=150)
        
        plt.close('all')
        
        logger.info(f"Saved visualizations to {viz_dir}")
        return {
            'nuclei': nuclei_viz_path,
            'cells': cell_viz_path,
            'combined': combined_viz_path
        }
    
    def extract_and_save_nuclei_centroids(self, nuclei_mask, well_id, file_id):
        """Extract nuclei centroids and save them.
        
        Args:
            nuclei_mask: Labeled nuclei mask
            well_id: Well identifier
            file_id: File identifier
            
        Returns:
            np.ndarray: Extracted centroids
        """
        # Extract centroids
        props = sm.regionprops(nuclei_mask)
        if not props:
            logger.warning(f"No nuclei found in {file_id}")
            return np.empty((0, 2))
            
        # Get centroids (y, x) format
        centroids = np.array([prop.centroid for prop in props])
        
        # Create well directory
        well_dir = os.path.join(self.output_dir, self.magnification, well_id)
        os.makedirs(well_dir, exist_ok=True)
        
        # Save local centroids
        local_centroids_file = os.path.join(well_dir, f"{file_id}_nuclei_centroids_local.npy")
        np.save(local_centroids_file, centroids)
        logger.info(f"Saved {len(centroids)} centroids to {local_centroids_file}")
        
        return centroids
    
    def run(self):
        """Run the segmentation pipeline.
        
        Returns:
            dict: Results summary
        """
        start_time = datetime.now()
        logger.info(f"Starting segmentation pipeline at {start_time}")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Extract file names for well and file IDs
        input_path = Path(self.input_file)
        input_fname = input_path.name
        file_id = input_path.stem
        
        # Parse well ID from filename (assuming Well1_Point1_... format)
        parts = input_fname.split('_')
        if parts[0].startswith('Well'):
            well_id = parts[0]
        else:
            well_id = 'Well1'  # Default if not found
            
        # Ensure the magnification directory exists (needed for benchmarking)
        mag_dir = os.path.join(self.output_dir, self.magnification)
        os.makedirs(mag_dir, exist_ok=True)
        
        try:
            # Load images
            nuclear_image, cell_image = self.load_nd2_images()
            
            # Segment nuclei and cells
            nuclei_mask = self.segment_nuclei(nuclear_image)
            cell_mask = self.segment_cells(cell_image, nuclear_image)
            
            # Clean masks
            clean_nuclei, clean_cells = self.clean_masks(nuclei_mask, cell_mask)
            
            # Create well directory
            well_dir = os.path.join(self.output_dir, self.magnification, well_id)
            os.makedirs(well_dir, exist_ok=True)
            
            # Save masks
            np.save(os.path.join(well_dir, f"{file_id}_nuclei_mask.npy"), clean_nuclei)
            np.save(os.path.join(well_dir, f"{file_id}_cell_mask.npy"), clean_cells)
            
            # Save visualization
            viz_paths = self.save_visualization(clean_nuclei, clean_cells, file_id)
            
            # Extract and save centroids
            centroids = self.extract_and_save_nuclei_centroids(
                clean_nuclei, well_id, file_id
            )
            
            # Create summary
            num_nuclei = len(np.unique(clean_nuclei)) - 1  # Subtract background
            num_cells = len(np.unique(clean_cells)) - 1    # Subtract background
            
            # Create properties
            properties = {
                'input_file': str(self.input_file),
                'nuclear_channel': self.nuclear_channel,
                'cell_channel': self.cell_channel,
                'well': well_id,
                'magnification': self.magnification,
                'num_nuclei': int(num_nuclei),
                'num_cells': int(num_cells),
                'processing_time': str(datetime.now() - start_time),
                'visualization': viz_paths
            }
            
            # Save properties
            with open(os.path.join(well_dir, f"{file_id}_properties.json"), 'w') as f:
                json.dump(properties, f, indent=2)
            
            logger.info(f"Segmentation completed: {num_cells} cells found, {len(centroids)} centroids extracted")
            
            # Return results
            return {
                "status": "success",
                "well": well_id,
                "file": file_id,
                "magnification": self.magnification,
                "num_nuclei": num_nuclei,
                "num_cells": num_cells,
                "num_centroids": len(centroids)
            }
            
        except Exception as e:
            logger.error(f"Segmentation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "well": well_id,
                "error": str(e)
            }


def run_nd2_segmentation(nd2_file, output_dir, magnification="10X", nuclear_channel=0, cell_channel=1):
    """Run the ND2 segmentation pipeline.
    
    Args:
        nd2_file: Path to ND2 file
        output_dir: Directory to save results
        magnification: "10X" or "40X"
        nuclear_channel: Index of nuclear channel (default: 0 for DAPI)
        cell_channel: Index of cell channel (default: 1)
        
    Returns:
        dict: Results summary
    """
    config = {
        'input_file': nd2_file,
        'output_dir': output_dir,
        'magnification': magnification,
        'nuclear_channel': nuclear_channel,
        'cell_channel': cell_channel,
        'save_visualization': True
    }
    
    segmentation = ND2Segmentation(config)
    results = segmentation.run()
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ND2 segmentation pipeline')
    parser.add_argument('--input', required=True, help='Path to ND2 file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--mag', choices=['10X', '40X'], default='10X', help='Magnification (10X or 40X)')
    parser.add_argument('--nuclear-channel', type=int, default=0, help='Index of nuclear channel (default: 0 for DAPI)')
    parser.add_argument('--cell-channel', type=int, default=1, help='Index of cell channel (default: 1)')
    
    args = parser.parse_args()
    
    run_nd2_segmentation(args.input, args.output, args.mag, args.nuclear_channel, args.cell_channel)