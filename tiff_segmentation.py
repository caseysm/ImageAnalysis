#!/usr/bin/env python3
"""
Segmentation script that works with TIFF files instead of ND2 files.
This is adapted from the original segmentation_10x.py and segmentation_40x.py.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime
import tifffile
from cellpose import models as cellpose_models
import skimage.measure as sm
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tiff-segmentation")

class TiffSegmentation:
    """
    Segmentation pipeline for TIFF images.
    Works with either 10X or 40X magnification.
    """
    
    def __init__(self, config):
        """Initialize the pipeline with configuration.
        
        Args:
            config (dict): Configuration parameters including:
                - input_dir: Directory containing TIFF files
                - output_dir: Directory to save results
                - nuclear_tiff: Path to nuclear channel TIFF
                - cell_tiff: Path to cell channel TIFF
                - magnification: "10X" or "40X"
                - save_visualization: Whether to save visualizations (default: True)
                - use_gpu: Whether to use GPU for cellpose (default: False)
        """
        self.config = config
        self.input_dir = config.get('input_dir')
        self.output_dir = config.get('output_dir')
        self.nuclear_tiff = config.get('nuclear_tiff')
        self.cell_tiff = config.get('cell_tiff')
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
    
    def load_tiff_images(self):
        """Load TIFF images for nuclear and cell channels.
        
        Returns:
            tuple: (nuclear_image, cell_image)
        """
        logger.info(f"Loading TIFF files:")
        logger.info(f"  Nuclear channel: {self.nuclear_tiff}")
        logger.info(f"  Cell channel: {self.cell_tiff}")
        
        try:
            # Load nuclear channel
            nuclear_image = tifffile.imread(self.nuclear_tiff)
            
            # Load cell channel
            cell_image = tifffile.imread(self.cell_tiff)
            
            logger.info(f"Nuclear image shape: {nuclear_image.shape}")
            logger.info(f"Cell image shape: {cell_image.shape}")
            
            return nuclear_image, cell_image
            
        except Exception as e:
            logger.error(f"Error loading TIFF files: {e}")
            raise
    
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
        nuclear_path = Path(self.nuclear_tiff)
        nuclear_fname = nuclear_path.name
        file_id = nuclear_path.stem

        # Parse well ID from filename (assuming Well1_Point1_... format)
        parts = nuclear_fname.split('_')
        if parts[0].startswith('Well'):
            well_id = parts[0]
        else:
            well_id = 'Well1'  # Default if not found

        # Ensure the magnification directory exists (needed for benchmarking)
        mag_dir = os.path.join(self.output_dir, self.magnification)
        os.makedirs(mag_dir, exist_ok=True)
        
        try:
            # Load images
            nuclear_image, cell_image = self.load_tiff_images()
            
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
                'nuclear_file': str(self.nuclear_tiff),
                'cell_file': str(self.cell_tiff),
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


def run_tiff_segmentation(nuclear_tiff, cell_tiff, output_dir, magnification="10X"):
    """Run the TIFF segmentation pipeline.
    
    Args:
        nuclear_tiff: Path to nuclear channel TIFF
        cell_tiff: Path to cell channel TIFF
        output_dir: Directory to save results
        magnification: "10X" or "40X"
        
    Returns:
        dict: Results summary
    """
    config = {
        'nuclear_tiff': nuclear_tiff,
        'cell_tiff': cell_tiff,
        'output_dir': output_dir,
        'magnification': magnification,
        'save_visualization': True
    }
    
    segmentation = TiffSegmentation(config)
    results = segmentation.run()
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run TIFF segmentation pipeline')
    parser.add_argument('--nuclear', required=True, help='Path to nuclear channel TIFF')
    parser.add_argument('--cell', required=True, help='Path to cell channel TIFF')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--mag', choices=['10X', '40X'], default='10X', help='Magnification (10X or 40X)')
    
    args = parser.parse_args()
    
    run_tiff_segmentation(args.nuclear, args.cell, args.output, args.mag)