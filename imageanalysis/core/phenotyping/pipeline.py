"""Phenotyping pipeline implementation based on original pipeline."""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
import nd2reader
import skimage.measure as sm
from scipy import ndimage as ndi
from typing import Dict, List, Tuple, Union, Optional

class PhenotypingPipeline:
    """Real phenotyping pipeline implementation based on original code.
    
    This pipeline extracts quantitative features from cell images including:
    - Morphological features (area, perimeter)
    - Intensity measurements (total, mean, std) across channels
    - Colocalization measurements between channels
    - Positional information
    """
    
    def __init__(self, config):
        """Initialize the pipeline with configuration.
        
        Args:
            config (dict): Configuration parameters including:
                - input_file: Path to the ND2 file with phenotyping images
                - segmentation_dir: Directory with segmentation results 
                - genotyping_dir: Directory with genotyping results (optional)
                - output_dir: Directory to save results
                - channels: List of channel names in the ND2 file (e.g. ['DAPI', 'GFP', 'RFP'])
                - wells: List of wells to process
                - colocalization_channels: Pair of channels to use for colocalization (e.g. [0, 1])
        """
        self.config = config
        self.input_file = config.get('input_file')
        self.segmentation_dir = config.get('segmentation_dir')
        self.genotyping_dir = config.get('genotyping_dir')
        self.output_dir = config.get('output_dir', os.path.join('results', 'phenotyping'))
        self.channels = config.get('channels', [])
        # Default channel names if not provided
        if not self.channels:
            self.channels = ['DAPI', 'GFP', 'RFP']
        self.wells = config.get('wells', [])
        self.colocalization_channels = config.get('colocalization_channels', [1, 2])  # Default to second and third channels
        self.logger = logging.getLogger("phenotyping")
        
    def load_nd2_image(self, file_path: str) -> np.ndarray:
        """Load an ND2 file and extract image data.
        
        Args:
            file_path: Path to the ND2 file
            
        Returns:
            Numpy array with dimensions (channels, height, width)
        """
        self.logger.info(f"Loading ND2 file: {file_path}")
        
        try:
            with nd2reader.ND2Reader(file_path) as images:
                # Get channels dimension
                channels = images.sizes['c']
                height = images.sizes['y']
                width = images.sizes['x']
                
                # Create array to hold all channels
                data = np.empty((channels, height, width), dtype=np.float32)
                
                # Load each channel
                for c in range(channels):
                    images.default_coords['c'] = c
                    data[c] = images[0]
                    
                return data
                
        except Exception as e:
            self.logger.error(f"Error loading ND2 file: {str(e)}")
            raise
            
    def load_segmentation_masks(self, well: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load nuclei and cell segmentation masks.
        
        Args:
            well: Well identifier (e.g., 'Well1')
            
        Returns:
            Tuple of (nuclei_mask, cell_mask)
        """
        self.logger.info(f"Loading segmentation masks for {well}")
        
        # Find masks for this well
        seg_dir = os.path.join(self.segmentation_dir, well)
        if not os.path.exists(seg_dir):
            raise FileNotFoundError(f"Segmentation directory not found: {seg_dir}")
            
        # Find nuclear and cell mask files
        nuclei_files = [f for f in os.listdir(seg_dir) if 'nuclei' in f.lower() and f.endswith('_mask.npy')]
        cell_files = [f for f in os.listdir(seg_dir) if 'cell' in f.lower() and 'nuclei' not in f.lower() and f.endswith('_mask.npy')]
        
        if not nuclei_files or not cell_files:
            raise FileNotFoundError(f"No mask files found for {well}")
            
        # Load the first matching files
        nuclei_mask = np.load(os.path.join(seg_dir, nuclei_files[0]))
        cell_mask = np.load(os.path.join(seg_dir, cell_files[0]))
        
        return nuclei_mask, cell_mask
    
    def load_genotyping_results(self, well: str) -> Optional[pd.DataFrame]:
        """Load genotyping results if available.
        
        Args:
            well: Well identifier (e.g., 'Well1')
            
        Returns:
            DataFrame with genotype assignments or None if not available
        """
        if not self.genotyping_dir:
            return None
            
        geno_file = os.path.join(self.genotyping_dir, well, f"{well}_genotypes.csv")
        if not os.path.exists(geno_file):
            self.logger.warning(f"No genotyping results found at {geno_file}")
            return None
            
        return pd.read_csv(geno_file)
    
    def crop_cell(self, 
                 image: np.ndarray, 
                 cell_mask: np.ndarray, 
                 cell_id: int, 
                 y: int, 
                 x: int) -> Tuple[np.ndarray, int, int, int, int]:
        """Crop a single cell from an image.
        
        Args:
            image: Multi-channel image with dimensions (channels, height, width)
            cell_mask: Cell segmentation mask with dimensions (height, width)
            cell_id: ID of the cell to crop
            y: Y-coordinate of the cell center
            x: X-coordinate of the cell center
            
        Returns:
            Tuple of (cropped_image, y_min, y_max, x_min, x_max)
        """
        # Get cell mask for this cell ID
        single_cell_mask = (cell_mask == cell_id)
        
        # Get projection of mask onto y and x axes
        y_proj = np.sum(single_cell_mask, axis=1) > 0
        x_proj = np.sum(single_cell_mask, axis=0) > 0
        
        # Find bounding box
        if not np.any(y_proj) or not np.any(x_proj):
            # Cell mask is empty
            return None, 0, 0, 0, 0
            
        y_min = np.argmax(y_proj)
        y_max = len(y_proj) - np.argmax(y_proj[::-1])
        
        x_min = np.argmax(x_proj)
        x_max = len(x_proj) - np.argmax(x_proj[::-1])
        
        # Expand box slightly
        padding = 5
        y_min = max(0, y_min - padding)
        y_max = min(cell_mask.shape[0], y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(cell_mask.shape[1], x_max + padding)
        
        # Extract crop for each channel
        # Create crop mask
        crop_mask = single_cell_mask[y_min:y_max, x_min:x_max]
        
        # Apply to each channel
        cropped_image = np.zeros((image.shape[0], y_max-y_min, x_max-x_min))
        
        for c in range(image.shape[0]):
            cropped_image[c] = image[c, y_min:y_max, x_min:x_max] * crop_mask
            
        return cropped_image, y_min, y_max, x_min, x_max
    
    def extract_features(self, 
                        cropped_image: np.ndarray,
                        nuclei_mask: np.ndarray,
                        cell_mask: np.ndarray,
                        cell_id: int,
                        y_min: int,
                        y_max: int,
                        x_min: int,
                        x_max: int) -> Dict:
        """Extract phenotypic features from a cropped cell.
        
        Args:
            cropped_image: Cropped cell image with dimensions (channels, height, width)
            nuclei_mask: Nuclei segmentation mask
            cell_mask: Cell segmentation mask
            cell_id: ID of the cell
            y_min, y_max, x_min, x_max: Bounding box coordinates
            
        Returns:
            Dictionary of phenotypic features
        """
        # Basic morphological features
        cell_props = sm.regionprops((cell_mask == cell_id).astype(int))
        nuclei_props = sm.regionprops((nuclei_mask == cell_id).astype(int))
        
        # Initialize feature dictionary
        features = {
            'cell_id': cell_id
        }
        
        # Add cell and nucleus properties if available
        if cell_props:
            features.update({
                'cell_area': cell_props[0].area,
                'cell_perimeter': cell_props[0].perimeter,
                'cell_eccentricity': cell_props[0].eccentricity,
                'cell_solidity': cell_props[0].solidity,
                'cell_extent': cell_props[0].extent,
                'cell_centroid_y': cell_props[0].centroid[0],
                'cell_centroid_x': cell_props[0].centroid[1]
            })
        else:
            features.update({
                'cell_area': 0,
                'cell_perimeter': 0,
                'cell_eccentricity': 0,
                'cell_solidity': 0,
                'cell_extent': 0,
                'cell_centroid_y': 0,
                'cell_centroid_x': 0
            })
            
        if nuclei_props:
            features.update({
                'nucleus_area': nuclei_props[0].area,
                'nucleus_perimeter': nuclei_props[0].perimeter,
                'nucleus_eccentricity': nuclei_props[0].eccentricity,
                'nucleus_solidity': nuclei_props[0].solidity,
                'nucleus_extent': nuclei_props[0].extent
            })
        else:
            features.update({
                'nucleus_area': 0,
                'nucleus_perimeter': 0,
                'nucleus_eccentricity': 0,
                'nucleus_solidity': 0,
                'nucleus_extent': 0
            })
            
        # Add intensity features for each channel
        for i, channel in enumerate(self.channels):
            if i < cropped_image.shape[0]:
                channel_data = cropped_image[i]
                # Remove zeros (background)
                channel_data_flat = channel_data.flatten()
                channel_data_nonzero = channel_data_flat[channel_data_flat > 0]
                
                if len(channel_data_nonzero) > 0:
                    features.update({
                        f'{channel}_total_intensity': np.sum(channel_data_nonzero),
                        f'{channel}_mean_intensity': np.mean(channel_data_nonzero),
                        f'{channel}_std_intensity': np.std(channel_data_nonzero),
                        f'{channel}_max_intensity': np.max(channel_data_nonzero),
                        f'{channel}_min_intensity': np.min(channel_data_nonzero)
                    })
                else:
                    features.update({
                        f'{channel}_total_intensity': 0,
                        f'{channel}_mean_intensity': 0,
                        f'{channel}_std_intensity': 0,
                        f'{channel}_max_intensity': 0,
                        f'{channel}_min_intensity': 0
                    })
        
        # Calculate nucleus-to-cell ratio
        if features['cell_area'] > 0:
            features['nucleus_to_cell_ratio'] = features['nucleus_area'] / features['cell_area']
        else:
            features['nucleus_to_cell_ratio'] = 0
            
        # Calculate circularity
        if features['cell_perimeter'] > 0:
            features['cell_circularity'] = (4 * np.pi * features['cell_area']) / (features['cell_perimeter'] ** 2)
        else:
            features['cell_circularity'] = 0
            
        # Add bounding box information
        features.update({
            'y_min': y_min,
            'y_max': y_max,
            'x_min': x_min,
            'x_max': x_max
        })
        
        # Calculate colocalization if possible
        if len(self.colocalization_channels) >= 2 and all(i < cropped_image.shape[0] for i in self.colocalization_channels):
            coloc_channels = self.colocalization_channels
            ch1_data = cropped_image[coloc_channels[0]].flatten()
            ch2_data = cropped_image[coloc_channels[1]].flatten()
            
            # Remove zeros and take only pixels where both channels have values
            nonzero_mask = (ch1_data > 0) & (ch2_data > 0)
            
            if np.sum(nonzero_mask) > 5:  # Ensure we have enough points
                ch1_nonzero = ch1_data[nonzero_mask]
                ch2_nonzero = ch2_data[nonzero_mask]
                
                # Calculate correlation coefficient
                if np.std(ch1_nonzero) > 0 and np.std(ch2_nonzero) > 0:
                    corr = np.corrcoef(ch1_nonzero, ch2_nonzero)[0, 1]
                    features['colocalization_r2'] = corr ** 2
                else:
                    features['colocalization_r2'] = 0
            else:
                features['colocalization_r2'] = 0
        else:
            features['colocalization_r2'] = 0
            
        return features
    
    def process_all_cells(self, 
                         image: np.ndarray, 
                         nuclei_mask: np.ndarray, 
                         cell_mask: np.ndarray,
                         genotypes: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Process all cells in an image.
        
        Args:
            image: Multi-channel image with dimensions (channels, height, width)
            nuclei_mask: Nuclei segmentation mask
            cell_mask: Cell segmentation mask
            genotypes: DataFrame with genotype assignments (optional)
            
        Returns:
            DataFrame with phenotypic features for all cells
        """
        # Get list of cell IDs
        cell_ids = np.unique(cell_mask)
        cell_ids = cell_ids[cell_ids > 0]  # Remove background (0)
        
        self.logger.info(f"Processing {len(cell_ids)} cells")
        
        all_features = []
        
        for cell_id in cell_ids:
            try:
                # Find centroid of cell
                cell_y, cell_x = ndi.center_of_mass(cell_mask == cell_id)
                
                # Crop cell
                cropped_image, y_min, y_max, x_min, x_max = self.crop_cell(
                    image, cell_mask, cell_id, int(cell_y), int(cell_x))
                
                if cropped_image is None:
                    self.logger.warning(f"Failed to crop cell {cell_id}")
                    continue
                    
                # Extract features
                features = self.extract_features(
                    cropped_image, nuclei_mask, cell_mask, cell_id, y_min, y_max, x_min, x_max)
                
                # Add genotype if available
                if genotypes is not None:
                    cell_genotype = genotypes[genotypes['cell_id'] == cell_id]
                    if not cell_genotype.empty:
                        if 'barcode' in cell_genotype.columns:
                            features['barcode'] = cell_genotype.iloc[0]['barcode']
                        if 'sgRNA' in cell_genotype.columns:
                            features['sgRNA'] = cell_genotype.iloc[0]['sgRNA']
                
                all_features.append(features)
                
            except Exception as e:
                self.logger.error(f"Error processing cell {cell_id}: {str(e)}")
                
        return pd.DataFrame(all_features)
    
    def run(self) -> Dict:
        """Run the phenotyping pipeline.
        
        Returns:
            Dictionary with results
        """
        self.logger.info("Starting phenotyping pipeline")
        start_time = datetime.now()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        try:
            # Get well name from input file
            input_path = Path(self.input_file)
            file_name = input_path.name
            well_match = file_name.split('_')[0]
            
            if well_match.startswith('Well'):
                well = well_match
            else:
                well = 'Well1'  # Default if not found
                
            # Skip if wells specified and not in list
            if self.wells and well not in self.wells:
                return {"status": "skipped", "well": well}
                
            # Create well directory
            well_dir = os.path.join(self.output_dir, well)
            os.makedirs(well_dir, exist_ok=True)
            
            # Load segmentation masks
            nuclei_mask, cell_mask = self.load_segmentation_masks(well)
            
            # Load genotyping results if available
            genotypes = self.load_genotyping_results(well)
            
            # Load image data
            image = self.load_nd2_image(self.input_file)
            
            # Process all cells
            phenotypes_df = self.process_all_cells(image, nuclei_mask, cell_mask, genotypes)
            
            # Save results
            if not phenotypes_df.empty:
                phenotypes_file = os.path.join(well_dir, f"{well}_phenotypes.csv")
                phenotypes_df.to_csv(phenotypes_file, index=False)
                
                # Create summary
                summary = {
                    'well': well,
                    'total_cells': len(np.unique(cell_mask)) - 1,  # Subtract 1 for background
                    'phenotyped_cells': len(phenotypes_df),
                    'channels': self.channels,
                    'processing_time': str(datetime.now() - start_time)
                }
                
                # Save summary
                summary_file = os.path.join(well_dir, "summary.json")
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                    
                return {
                    "status": "success",
                    "well": well,
                    "cells_processed": summary['total_cells'],
                    "phenotyped_cells": summary['phenotyped_cells'],
                    "processing_time": summary['processing_time']
                }
            else:
                return {
                    "status": "error",
                    "well": well,
                    "message": "No cells were successfully phenotyped"
                }
                
        except Exception as e:
            self.logger.error(f"Error in phenotyping pipeline: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }