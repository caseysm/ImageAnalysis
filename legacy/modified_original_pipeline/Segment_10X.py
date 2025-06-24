from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import nd2reader as nd2
from cellpose import models as cellpose_models
import skimage.measure as sm
import os
from os import listdir, makedirs
from os.path import isfile, join, dirname, exists
import sys

# Add parent directory to import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from original pipeline
from original_pipeline import In_Situ_Functions as isf
import warnings
warnings.filterwarnings('ignore')


def segment_nuclei(img, nuc_diameter=30):
    """Segment nuclei using cellpose."""
    # Use cellpose to segment nuclei
    model = cellpose_models.Cellpose(gpu=False, model_type='nuclei')
    masks, _, _, _ = model.eval(img, diameter=nuc_diameter, channels=[[0, 0]])
    return masks


def segment_cells(img, NUC=None, cell_diameter=60):
    """Segment cells using cellpose."""
    # Use cellpose to segment cells
    model = cellpose_models.Cellpose(gpu=False, model_type='cyto2')
    
    if NUC is not None:
        # Create a 2-channel image with cell stain and nuclear stain
        # Following cellpose recommended format
        img_combined = np.zeros((img.shape[0], img.shape[1], 2))
        img_combined[:, :, 0] = img
        img_combined[:, :, 1] = NUC
        masks, _, _, _ = model.eval(img_combined, diameter=cell_diameter, channels=[[1, 2]])
    else:
        masks, _, _, _ = model.eval(img, diameter=cell_diameter, channels=[[0, 0]])
    
    return masks


def label_and_clean(nucs, cells):
    """Clean up segmentation and ensure consistent labeling."""
    # Get region properties
    props_nucs = sm.regionprops(nucs)
    props_cells = sm.regionprops(cells)
    
    # Extract centroids
    centroids_nucs = np.array([p.centroid for p in props_nucs])
    
    # Initialize cleaned masks
    clean_nucs = np.zeros_like(nucs)
    clean_cells = np.zeros_like(cells)
    
    # For each nucleus, find corresponding cell and use consistent labeling
    cell_id = 1
    for i, (y, x) in enumerate(centroids_nucs):
        nuc_label = i + 1
        
        # Skip small nuclei
        if props_nucs[i].area < 100:
            continue
        
        # Get coordinates to check in cell mask
        y_int, x_int = int(y), int(x)
        
        # Check if coordinates are within bounds
        if 0 <= y_int < cells.shape[0] and 0 <= x_int < cells.shape[1]:
            # Get cell label at nucleus position
            cell_label = cells[y_int, x_int]
            
            # Only include cells with exactly one nucleus
            if cell_label > 0:
                # Count nuclei in this cell
                nuc_count = 0
                for ny, nx in centroids_nucs:
                    ny_int, nx_int = int(ny), int(nx)
                    if 0 <= ny_int < cells.shape[0] and 0 <= nx_int < cells.shape[1]:
                        if cells[ny_int, nx_int] == cell_label:
                            nuc_count += 1
                
                # Only keep cells with exactly one nucleus
                if nuc_count == 1:
                    # Copy with consistent labeling
                    clean_nucs[nucs == nuc_label] = cell_id
                    clean_cells[cells == cell_label] = cell_id
                    cell_id += 1
    
    return clean_nucs, clean_cells


if __name__ == '__main__':
    t_start = datetime.now()

    parser = argparse.ArgumentParser(description="Segment cells and nuclei from ND2 files")
    parser.add_argument('input_file', help='Path to input ND2 file')
    parser.add_argument('--output-dir', help='Output directory', default='segmented/10X')
    parser.add_argument('--nuc-channel', type=int, default=0, help='Nuclear channel index (default: 0)')
    parser.add_argument('--cell-channel', type=int, default=1, help='Cell channel index (default: 1)')
    args = parser.parse_args()

    try:
        # Extract well and tile info from filename
        input_path = args.input_file
        filename = os.path.basename(input_path)
        parts = filename.split('_')
        
        # Try to extract well info (assuming Well1_Point1_... format)
        if parts[0].startswith('Well'):
            well_str = parts[0]
            well_num = int(well_str.replace('Well', ''))
        else:
            well_str = 'Well1'
            well_num = 1
            
        # Try to extract tile/point info
        if len(parts) > 1 and parts[1].startswith('Point'):
            tile_str = parts[1]
            tile_num = int(tile_str.replace('Point', ''))
        else:
            # Try to get from sequence number
            seq_match = next((p for p in parts if p.startswith('Seq')), None)
            if seq_match:
                tile_num = int(seq_match.replace('Seq', ''))
            else:
                tile_num = 1
                
        print(f"Processing {filename} - Well: {well_num}, Tile: {tile_num}")
        
        # Load ND2 file
        with nd2.ND2Reader(input_path) as images:
            # Get dimensions
            channels = images.sizes.get('c', 1)
            
            # Load nuclear and cell channels
            images.default_coords['c'] = args.nuc_channel
            nuc_img = images[0]
            
            if channels > args.cell_channel:
                images.default_coords['c'] = args.cell_channel
                cell_img = images[0]
            else:
                cell_img = nuc_img  # Use nuclear channel if cell channel not available
                
        # Segment nuclei and cells
        print("Segmenting nuclei...")
        nucs = segment_nuclei(nuc_img, nuc_diameter=30)
        
        print("Segmenting cells...")
        cells = segment_cells(cell_img, NUC=nuc_img, cell_diameter=60)
        
        print("Cleaning and labeling...")
        nucs, cells = label_and_clean(nucs, cells)
        
        # Create output directory structure
        output_dir = args.output_dir
        nuc_dir = os.path.join(output_dir, "nucs", f"well_{well_num}")
        cell_dir = os.path.join(output_dir, "cells", f"well_{well_num}")
        
        os.makedirs(nuc_dir, exist_ok=True)
        os.makedirs(cell_dir, exist_ok=True)
        
        # Save segmentation results in original format 
        save_name_nuc = os.path.join(nuc_dir, f"Seg_Nuc-Well_{well_num}_Tile_{tile_num}.npy")
        save_name_cell = os.path.join(cell_dir, f"Seg_Cells-Well_{well_num}_Tile_{tile_num}.npy")
        
        # Create a tuple with image and mask, to match original format
        np.save(save_name_nuc, (nuc_img, nucs))
        np.save(save_name_cell, (cell_img, cells))
        
        # Also save to the new script output format if specified
        if args.output_dir != 'segmented/10X':
            base_name = os.path.splitext(filename)[0]
            np.save(os.path.join(args.output_dir, f"{base_name}_nuclei_mask.npy"), nucs)
            np.save(os.path.join(args.output_dir, f"{base_name}_cell_mask.npy"), cells)
            
            # Save properties
            props = {
                'file': input_path,
                'well': well_str,
                'tile': tile_num,
                'num_nuclei': int(np.max(nucs)),
                'num_cells': int(np.max(cells))
            }
            
            import json
            with open(os.path.join(args.output_dir, f"{base_name}_properties.json"), 'w') as f:
                json.dump(props, f, indent=2)
            
        print(f"Segmentation completed: {np.max(nucs)} nuclei, {np.max(cells)} cells")
        
        t_end = datetime.now()
        print(f'Well: {well_num} Tile: {tile_num} Time Start: {t_start} Time End: {t_end} Duration: {t_end - t_start}')
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)