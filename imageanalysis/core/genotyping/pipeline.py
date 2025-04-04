"""Genotyping pipeline implementation based on original pipeline."""

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

class StandardGenotypingPipeline:
    """Standard genotyping pipeline implementation.
    
    This pipeline implements barcode calling and assignment based on the original
    In_Situ_Functions implementation. It processes fluorescence images to identify
    genetic barcodes and assigns them to segmented cells.
    """
    
    def __init__(self, config):
        """Initialize the pipeline with configuration.
        
        Args:
            config (dict): Configuration parameters including:
                - input_file: Path to the ND2 file or directory with cycle files
                - segmentation_dir: Directory with segmentation results
                - barcode_library: Path to CSV file with barcode library
                - output_dir: Directory to save results
                - wells: List of wells to process
                - peak_threshold: Threshold for peak detection (default: 200)
                - min_quality_score: Minimum quality score for barcode (default: 0.3)
                - max_hamming_distance: Maximum Hamming distance for barcode matching (default: 1)
        """
        self.config = config
        self.input_file = config.get('input_file')
        self.segmentation_dir = config.get('segmentation_dir')
        self.barcode_library = config.get('barcode_library')
        self.output_dir = config.get('output_dir', os.path.join('results', 'genotyping'))
        self.wells = config.get('wells', [])
        self.peak_threshold = config.get('peak_threshold', 200)
        self.min_quality_score = config.get('min_quality_score', 0.3)
        self.max_hamming_distance = config.get('max_hamming_distance', 1)
        self.logger = logging.getLogger("genotyping")
        
    def load_nd2_cycle_data(self, cycle_dir: str, well_id: int, tile_id: int) -> np.ndarray:
        """Load ND2 files from a cycle directory for a specific well and tile.
        
        Args:
            cycle_dir: Path to the cycle directory
            well_id: Well identifier
            tile_id: Tile identifier
            
        Returns:
            Numpy array with dimensions (channels, height, width)
        """
        self.logger.info(f"Loading ND2 data from {cycle_dir} for well {well_id}, tile {tile_id}")
        
        # Find ND2 files matching the well and tile pattern
        cycle_path = Path(cycle_dir)
        files = [f for f in os.listdir(cycle_path) if f.endswith('.nd2')]
        
        matching_file = None
        for file in files:
            parts = file.split('_')
            try:
                file_well = int(parts[0].replace('Well', ''))
                file_tile = int(parts[2])
                
                if file_well == well_id and file_tile == tile_id:
                    matching_file = file
                    break
            except (IndexError, ValueError):
                continue
                
        if matching_file is None:
            raise FileNotFoundError(f"No matching ND2 file found for well {well_id}, tile {tile_id} in {cycle_dir}")
            
        # Load the ND2 file
        file_path = cycle_path / matching_file
        with nd2reader.ND2Reader(str(file_path)) as nd2_data:
            channels = nd2_data.sizes['c']
            height = nd2_data.sizes['y']
            width = nd2_data.sizes['x']
            
            # Load all channels
            data = np.empty((channels, height, width), dtype=np.float64)
            for c in range(channels):
                nd2_data.default_coords['c'] = c
                data[c] = nd2_data[0]
                
        return data
        
    def assemble_cycle_data(self, well_id: int, tile_id: int) -> np.ndarray:
        """Assemble data from multiple cycle directories.
        
        Args:
            well_id: Well identifier
            tile_id: Tile identifier
            
        Returns:
            Numpy array with dimensions (cycles, channels, height, width)
        """
        self.logger.info(f"Assembling cycle data for well {well_id}, tile {tile_id}")
        
        # Get base directory for genotyping data
        if isinstance(self.input_file, str) and os.path.isdir(self.input_file):
            base_dir = Path(self.input_file)
        else:
            base_dir = Path(self.input_file).parent
            
        # Find cycle directories
        cycle_dirs = [d for d in os.listdir(base_dir) 
                     if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('cycle_')]
        cycle_dirs = sorted(cycle_dirs, key=lambda x: int(x.split('_')[1]))
        
        if not cycle_dirs:
            raise ValueError(f"No cycle directories found in {base_dir}")
            
        cycle_data = []
        for cycle_dir in cycle_dirs:
            try:
                data = self.load_nd2_cycle_data(os.path.join(base_dir, cycle_dir), well_id, tile_id)
                cycle_data.append(data)
            except FileNotFoundError:
                self.logger.warning(f"No data found for well {well_id}, tile {tile_id} in {cycle_dir}")
                
        if not cycle_data:
            raise ValueError(f"No data loaded for well {well_id}, tile {tile_id}")
            
        # Stack cycle data
        return np.stack(cycle_data)
        
    def align_cycles(self, data: np.ndarray) -> np.ndarray:
        """Align cycles using DAPI as a reference channel.
        
        Args:
            data: Cycle data with dimensions (cycles, channels, height, width)
            
        Returns:
            Aligned cycle data
        """
        self.logger.info("Aligning cycles")
        
        # Assuming first channel is DAPI
        reference_channel = 0
        
        # Initialize aligned data with the first cycle
        aligned_data = np.copy(data)
        reference = data[0, reference_channel]
        
        # Align each subsequent cycle to the first
        for c in range(1, data.shape[0]):
            # Calculate shift between DAPI channels
            shift, _ = self._calculate_shift(reference, data[c, reference_channel])
            
            # Apply shift to all channels in this cycle
            for ch in range(data.shape[1]):
                aligned_data[c, ch] = self._shift_image(data[c, ch], shift)
                
        return aligned_data
        
    def _calculate_shift(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[Tuple[int, int], float]:
        """Calculate shift between two images using phase correlation.
        
        Args:
            image1: Reference image
            image2: Image to align
            
        Returns:
            Tuple of (y_shift, x_shift) and correlation value
        """
        # Normalize images
        img1 = image1.astype(np.float32) - np.mean(image1)
        img2 = image2.astype(np.float32) - np.mean(image2)
        
        # Compute FFT
        f1 = np.fft.fft2(img1)
        f2 = np.fft.fft2(img2)
        
        # Compute cross-power spectrum
        cross_power = f1 * np.conj(f2)
        cross_power /= np.abs(cross_power)
        
        # Compute inverse FFT and find maximum
        inverse_fft = np.fft.ifft2(cross_power).real
        y_max, x_max = np.unravel_index(np.argmax(inverse_fft), inverse_fft.shape)
        
        # Adjust shift for possible wraparound
        y_shift = y_max if y_max < img1.shape[0] // 2 else y_max - img1.shape[0]
        x_shift = x_max if x_max < img1.shape[1] // 2 else x_max - img1.shape[1]
        
        return (y_shift, x_shift), inverse_fft[y_max, x_max]
        
    def _shift_image(self, image: np.ndarray, shift: Tuple[int, int]) -> np.ndarray:
        """Shift image by given amount.
        
        Args:
            image: Image to shift
            shift: (y_shift, x_shift) tuple
            
        Returns:
            Shifted image
        """
        return ndi.shift(image, shift, mode='constant', cval=0)
        
    def transform_log(self, data: np.ndarray, skip_index: int = 0) -> np.ndarray:
        """Apply Laplacian of Gaussian filter to enhance spots.
        
        Args:
            data: Aligned cycle data with dimensions (cycles, channels, height, width)
            skip_index: Index of channel to skip (usually DAPI)
            
        Returns:
            LoG-filtered data
        """
        self.logger.info("Applying LoG filter")
        
        # Create copy of data
        loged_data = np.copy(data)
        
        # Apply LoG filter to all channels except skip_index
        for c in range(data.shape[0]):  # Cycles
            for ch in range(data.shape[1]):  # Channels
                if ch != skip_index:
                    loged_data[c, ch] = ndi.gaussian_laplace(data[c, ch], sigma=1)
                    
        return loged_data
        
    def max_filter(self, data: np.ndarray, width: int = 3, skip_index: int = 0) -> np.ndarray:
        """Apply maximum filter to enhance spots.
        
        Args:
            data: LoG-filtered data
            width: Width of maximum filter
            skip_index: Index of channel to skip (usually DAPI)
            
        Returns:
            Max-filtered data
        """
        self.logger.info("Applying max filter")
        
        # Create copy of data
        maxed_data = np.copy(data)
        
        # Apply max filter to all channels except skip_index
        for c in range(data.shape[0]):  # Cycles
            for ch in range(data.shape[1]):  # Channels
                if ch != skip_index:
                    maxed_data[c, ch] = ndi.maximum_filter(data[c, ch], size=width)
                    
        return maxed_data
        
    def compute_std(self, data: np.ndarray, skip_index: int = 0) -> np.ndarray:
        """Compute standard deviation over cycles.
        
        Args:
            data: LoG-filtered data
            skip_index: Index of channel to skip (usually DAPI)
            
        Returns:
            Standard deviation image
        """
        self.logger.info("Computing standard deviation")
        
        # Remove skip_index channel
        channels_mask = np.ones(data.shape[1], dtype=bool)
        channels_mask[skip_index] = False
        filtered_data = data[:, channels_mask]
        
        # Compute standard deviation over cycles
        std_data = np.std(filtered_data, axis=0).mean(axis=0)
        
        return std_data
        
    def find_peaks(self, std_data: np.ndarray, threshold: float) -> np.ndarray:
        """Find peaks in the standard deviation image.
        
        Args:
            std_data: Standard deviation image
            threshold: Threshold for peak detection
            
        Returns:
            Binary mask of peaks
        """
        self.logger.info(f"Finding peaks with threshold {threshold}")
        
        # Find local maxima
        maxima = ndi.maximum_filter(std_data, size=5)
        peaks = (std_data == maxima) & (std_data > threshold)
        
        return peaks
        
    def call_bases(self, cell_mask: np.ndarray, maxed_data: np.ndarray, peaks: np.ndarray) -> pd.DataFrame:
        """Call bases from peaks.
        
        Args:
            cell_mask: Cell segmentation mask
            maxed_data: Max-filtered data
            peaks: Binary mask of peaks
            
        Returns:
            DataFrame with barcode calls
        """
        self.logger.info("Calling bases")
        
        # Extract nucleotide data for each peak
        peak_coords = np.where(peaks)
        n_peaks = len(peak_coords[0])
        
        if n_peaks == 0:
            self.logger.warning("No peaks found")
            return pd.DataFrame()
            
        reads = []
        for i in range(n_peaks):
            y, x = peak_coords[0][i], peak_coords[1][i]
            
            # Get cell ID
            cell_id = cell_mask[y, x]
            
            # Skip if not in a cell
            if cell_id == 0:
                continue
                
            # Extract intensities for all cycles and channels
            n_cycles = maxed_data.shape[0]
            n_channels = maxed_data.shape[1] - 1  # Excluding DAPI
            
            # Assuming channels 1-4 are G, T, A, C (index 1-4, excluding DAPI at index 0)
            intensities = np.zeros((n_cycles, n_channels))
            for c in range(n_cycles):
                for ch in range(n_channels):
                    intensities[c, ch] = maxed_data[c, ch+1, y, x]
                    
            # Call base for each cycle
            bases = []
            quality_scores = []
            
            for c in range(n_cycles):
                cycle_intensities = intensities[c]
                max_idx = np.argmax(cycle_intensities)
                
                # Get sorted intensities
                sorted_intensities = np.sort(cycle_intensities)
                
                # Calculate quality score (ratio of top to second intensity)
                if sorted_intensities[-1] > 0:
                    q_score = 1.0 - (sorted_intensities[-2] / sorted_intensities[-1])
                else:
                    q_score = 0.0
                    
                quality_scores.append(q_score)
                
                # Convert index to base (G=0, T=1, A=2, C=3)
                bases.append(['G', 'T', 'A', 'C'][max_idx])
                
            # Combine into barcode
            barcode = ''.join(bases)
            
            # Create read entry
            read = {
                'cell': int(cell_id),
                'barcode': barcode,
                'i': y,
                'j': x,
                'peak_value': peaks[y, x]
            }
            
            # Add quality scores
            for c in range(n_cycles):
                read[f'Q_{c}'] = quality_scores[c]
                
            reads.append(read)
            
        return pd.DataFrame(reads)
        
    def assign_ambiguity(self, reads_df: pd.DataFrame, min_quality: float = 0.3) -> pd.DataFrame:
        """Assign ambiguity codes to low-quality base calls.
        
        Args:
            reads_df: DataFrame with barcode calls
            min_quality: Minimum quality score for a definite base call
            
        Returns:
            DataFrame with ambiguity codes
        """
        self.logger.info(f"Assigning ambiguity codes with min quality {min_quality}")
        
        # Create a copy of the DataFrame
        df_amb = reads_df.copy()
        
        if df_amb.empty:
            return df_amb
            
        # Get quality score columns
        q_cols = [col for col in df_amb.columns if col.startswith('Q_')]
        n_cycles = len(q_cols)
        
        # Process each read
        for idx, row in df_amb.iterrows():
            barcode = list(row['barcode'])
            
            # Check quality scores for each position
            for i in range(n_cycles):
                q_score = row[f'Q_{i}']
                
                # If quality is too low, replace with N
                if q_score < min_quality:
                    barcode[i] = 'N'
                    
            # Update barcode
            df_amb.at[idx, 'barcode'] = ''.join(barcode)
            
        return df_amb
        
    def match_barcodes(self, reads_df: pd.DataFrame, library_df: pd.DataFrame) -> pd.DataFrame:
        """Match barcodes to the library.
        
        Args:
            reads_df: DataFrame with barcode calls
            library_df: DataFrame with barcode library
            
        Returns:
            DataFrame with matched barcodes
        """
        self.logger.info("Matching barcodes to library")
        
        if reads_df.empty:
            return pd.DataFrame()
            
        # Extract barcode column from library
        library_barcodes = library_df['sgRNA_seq'].values
        n_cycles = len(reads_df['barcode'].iloc[0])
        
        # Truncate library barcodes to the number of cycles used
        library_barcodes = [b[:n_cycles] for b in library_barcodes]
        
        # Group reads by cell
        cell_groups = reads_df.groupby('cell')
        
        matches = []
        for cell_id, cell_df in cell_groups:
            # Count barcodes in this cell
            barcode_counts = cell_df['barcode'].value_counts()
            
            best_match = None
            min_distance = n_cycles + 1
            
            # For each barcode in the cell
            for barcode, count in barcode_counts.items():
                # Skip barcodes with N
                if 'N' in barcode:
                    continue
                    
                # Compare to each library barcode
                for lib_idx, lib_barcode in enumerate(library_barcodes):
                    # Calculate Hamming distance
                    distance = sum(a != b for a, b in zip(barcode, lib_barcode))
                    
                    # If better match, update
                    if distance < min_distance and distance <= self.max_hamming_distance:
                        min_distance = distance
                        best_match = {
                            'cell_id': cell_id,
                            'barcode': barcode,
                            'sgRNA': library_df['sgRNA'].values[lib_idx],
                            'sgRNA_seq': library_df['sgRNA_seq'].values[lib_idx][:n_cycles],
                            'hamming_distance': distance,
                            'read_count': count
                        }
                        
            # If a match was found, add to results
            if best_match is not None:
                matches.append(best_match)
                
        return pd.DataFrame(matches)
        
    def load_segmentation_masks(self, well: str, tile_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load segmentation masks for a specific well and tile.
        
        Args:
            well: Well identifier (e.g., 'Well1')
            tile_id: Tile identifier
            
        Returns:
            Tuple of (nuclei_mask, cell_mask)
        """
        self.logger.info(f"Loading segmentation masks for {well}, tile {tile_id}")
        
        # Find masks for this well
        seg_dir = os.path.join(self.segmentation_dir, well)
        if not os.path.exists(seg_dir):
            raise FileNotFoundError(f"Segmentation directory not found: {seg_dir}")
            
        # Find nuclear and cell mask files
        mask_files = [f for f in os.listdir(seg_dir) 
                     if f.endswith('_mask.npy') or f.endswith('_masks.npy')]
        
        # Find files that match the tile ID
        tile_files = []
        for f in mask_files:
            parts = f.split('_')
            for part in parts:
                if part.isdigit() and int(part) == tile_id:
                    tile_files.append(f)
                    break
                    
        if not tile_files:
            # If no specific tile files found, use the first available masks
            nuclei_files = [f for f in mask_files if 'nuclei' in f.lower()]
            cell_files = [f for f in mask_files if 'cell' in f.lower() and 'nuclei' not in f.lower()]
            
            if not nuclei_files or not cell_files:
                raise FileNotFoundError(f"No mask files found for {well}, tile {tile_id}")
                
            nuclei_mask = np.load(os.path.join(seg_dir, nuclei_files[0]))
            cell_mask = np.load(os.path.join(seg_dir, cell_files[0]))
        else:
            # Use tile-specific masks
            nuclei_files = [f for f in tile_files if 'nuclei' in f.lower()]
            cell_files = [f for f in tile_files if 'cell' in f.lower() and 'nuclei' not in f.lower()]
            
            if not nuclei_files or not cell_files:
                raise FileNotFoundError(f"No mask files found for {well}, tile {tile_id}")
                
            nuclei_mask = np.load(os.path.join(seg_dir, nuclei_files[0]))
            cell_mask = np.load(os.path.join(seg_dir, cell_files[0]))
            
        return nuclei_mask, cell_mask
        
    def load_barcode_library(self) -> pd.DataFrame:
        """Load barcode library from CSV file.
        
        Returns:
            DataFrame with barcode library
        """
        self.logger.info(f"Loading barcode library from {self.barcode_library}")
        
        if not os.path.exists(self.barcode_library):
            raise FileNotFoundError(f"Barcode library not found: {self.barcode_library}")
            
        return pd.read_csv(self.barcode_library)
        
    def process_well_tile(self, well: str, well_id: int, tile_id: int) -> Dict:
        """Process a specific well and tile.
        
        Args:
            well: Well identifier (e.g., 'Well1')
            well_id: Well number
            tile_id: Tile identifier
            
        Returns:
            Dictionary with results
        """
        try:
            # Load segmentation masks
            nuclei_mask, cell_mask = self.load_segmentation_masks(well, tile_id)
            
            # Assemble data from all cycles
            data = self.assemble_cycle_data(well_id, tile_id)
            
            # Align cycles
            aligned_data = self.align_cycles(data)
            
            # Apply LoG filter
            loged_data = self.transform_log(aligned_data)
            
            # Apply max filter
            maxed_data = self.max_filter(loged_data)
            
            # Compute standard deviation
            std_data = self.compute_std(loged_data)
            
            # Find peaks
            peaks = self.find_peaks(std_data, self.peak_threshold)
            
            # Call bases
            reads_df = self.call_bases(cell_mask, maxed_data, peaks)
            
            # Assign ambiguity
            reads_amb_df = self.assign_ambiguity(reads_df, self.min_quality_score)
            
            # Load barcode library
            library_df = self.load_barcode_library()
            
            # Match barcodes
            genotypes_df = self.match_barcodes(reads_amb_df, library_df)
            
            # Save results
            well_dir = os.path.join(self.output_dir, well)
            os.makedirs(well_dir, exist_ok=True)
            
            # Create file names
            base_name = f"{well}_Tile{tile_id}"
            reads_file = os.path.join(well_dir, f"{base_name}_reads.csv")
            reads_amb_file = os.path.join(well_dir, f"{base_name}_reads_amb.csv")
            genotypes_file = os.path.join(well_dir, f"{base_name}_genotypes.csv")
            
            # Save files
            if not reads_df.empty:
                reads_df.to_csv(reads_file, index=False)
            
            if not reads_amb_df.empty:
                reads_amb_df.to_csv(reads_amb_file, index=False)
                
            if not genotypes_df.empty:
                genotypes_df.to_csv(genotypes_file, index=False)
                
            # Create summary
            summary = {
                'well': well,
                'tile': tile_id,
                'total_cells': int(np.max(cell_mask)),
                'total_reads': len(reads_df),
                'assigned_cells': len(genotypes_df),
                'unique_barcodes': len(genotypes_df['sgRNA'].unique()) if not genotypes_df.empty else 0
            }
            
            return {
                "status": "success",
                **summary
            }
            
        except Exception as e:
            self.logger.error(f"Error processing {well}, tile {tile_id}: {str(e)}")
            return {
                "status": "error",
                "well": well,
                "tile": tile_id,
                "error": str(e)
            }
            
    def combine_tile_results(self, well: str, results: List[Dict]) -> None:
        """Combine results from multiple tiles for a well.
        
        Args:
            well: Well identifier
            results: List of tile processing results
        """
        self.logger.info(f"Combining tile results for {well}")
        
        well_dir = os.path.join(self.output_dir, well)
        
        # Find all genotype files
        genotype_files = [f for f in os.listdir(well_dir) 
                         if f.endswith('_genotypes.csv')]
        
        # Load and combine
        all_genotypes = []
        for file in genotype_files:
            file_path = os.path.join(well_dir, file)
            try:
                df = pd.read_csv(file_path)
                if not df.empty:
                    all_genotypes.append(df)
            except Exception as e:
                self.logger.warning(f"Error loading {file_path}: {str(e)}")
                
        # If no genotypes, return
        if not all_genotypes:
            self.logger.warning(f"No genotype data found for {well}")
            return
            
        # Combine and save
        combined_df = pd.concat(all_genotypes)
        combined_file = os.path.join(well_dir, f"{well}_genotypes.csv")
        combined_df.to_csv(combined_file, index=False)
        
        # Create summary
        successful_tiles = [r for r in results if r["status"] == "success"]
        
        summary = {
            'well': well,
            'total_cells': sum(r.get('total_cells', 0) for r in successful_tiles),
            'total_reads': sum(r.get('total_reads', 0) for r in successful_tiles),
            'assigned_cells': len(combined_df),
            'unique_barcodes': len(combined_df['sgRNA'].unique()),
            'tiles_processed': len(successful_tiles),
            'processing_time': str(datetime.now())
        }
        
        # Save summary
        summary_file = os.path.join(well_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
    def run(self) -> Dict:
        """Run the genotyping pipeline.
        
        Returns:
            Dictionary with results
        """
        self.logger.info("Starting genotyping pipeline")
        start_time = datetime.now()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get or extract well name from input file
        input_path = Path(self.input_file)
        file_name = input_path.name
        well_match = file_name.split('_')[0]
        
        if well_match.startswith('Well'):
            well = well_match
            well_id = int(well.replace('Well', ''))
        else:
            well = 'Well1'  # Default if not found
            well_id = 1
            
        # Skip if wells specified and not in list
        if self.wells and well not in self.wells:
            return {"status": "skipped", "well": well}
            
        # Find tiles to process
        self.logger.info(f"Processing well {well}")
        
        # Get segmentation directory for this well
        seg_dir = os.path.join(self.segmentation_dir, well)
        if not os.path.exists(seg_dir):
            return {"status": "error", "message": f"No segmentation results found for well {well}"}
            
        # Find tiles to process based on segmentation files
        mask_files = [f for f in os.listdir(seg_dir) 
                     if f.endswith('_mask.npy') or f.endswith('_masks.npy')]
        
        tiles = set()
        for file in mask_files:
            parts = file.split('_')
            for part in parts:
                if part.isdigit():
                    tiles.add(int(part))
                    break
                    
        if not tiles:
            # Default to tile 1
            tiles = {1}
            
        # Process each tile
        tile_results = []
        for tile_id in sorted(tiles):
            result = self.process_well_tile(well, well_id, tile_id)
            tile_results.append(result)
            
        # Combine results
        if any(r["status"] == "success" for r in tile_results):
            self.combine_tile_results(well, tile_results)
            
        # Create overall results
        successful_tiles = [r for r in tile_results if r["status"] == "success"]
        error_tiles = [r for r in tile_results if r["status"] == "error"]
        
        result = {
            "status": "success" if successful_tiles else "error",
            "well": well,
            "tiles_processed": len(successful_tiles),
            "tiles_failed": len(error_tiles),
            "processing_time": str(datetime.now() - start_time)
        }
        
        if successful_tiles:
            result.update({
                "total_cells": sum(r.get('total_cells', 0) for r in successful_tiles),
                "total_reads": sum(r.get('total_reads', 0) for r in successful_tiles),
                "assigned_cells": sum(r.get('assigned_cells', 0) for r in successful_tiles)
            })
            
        self.logger.info(f"Genotyping pipeline completed for {well}")
        return result