from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import os
import sys
import nd2reader as nd2
from cellpose import models as cellpose_models
import skimage.measure as sm
from os import listdir
from os.path import isfile, join
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from modified pipeline
from modified_original_pipeline import In_Situ_Functions as isf


if __name__ == '__main__':
    t_start = datetime.now()

    # Take job number from shell script, requires 5 digits in the form: well (_), tile (_,_,_,_)
    # Example well 2 tile 348 to be inputed as 20348
    # This allows running the segmentation on all tiles in parallel
    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=int)
    parser.add_argument('--input-dir', type=str, default='phenotyping', 
                        help='Directory containing ND2 files')
    parser.add_argument('--output-dir', type=str, default='segmented/40X', 
                        help='Base directory for output files')
    parser.add_argument('--nuc-diameter', type=int, default=120, 
                        help='Diameter parameter for nuclei segmentation')
    parser.add_argument('--cell-diameter', type=int, default=240, 
                        help='Diameter parameter for cell segmentation')
    parser.add_argument('--gpu', action='store_true', 
                        help='Use GPU for segmentation')
    
    args = parser.parse_args()
    n_job = args.job
    n_well = int(np.floor(n_job / 10000))
    n_tile = int(np.around(10000 * ((n_job / 10000) % 1)))

    # Raw phenotyping data to be put in a dir called 'phenotyping'. All wells and tiles can be in the same dir
    path_name = args.input_dir
    img = isf.InSitu.Import_ND2_by_Tile_and_Well(n_tile, n_well, path_name)

    # Segmentation lines using cellpose, defining diameter encouraged
    # nucs and cells are segmentation mask arrays
    # The DAPI nuclear channel is 0, and the cellular segmentation dye channel is 1.
    print(f"Segmenting nuclei for well {n_well}, tile {n_tile}...")
    nucs = isf.Segment.Segment_Nuclei(img[0], nuc_diameter=args.nuc_diameter, GPU=args.gpu)
    
    print(f"Segmenting cells for well {n_well}, tile {n_tile}...")
    cells = isf.Segment.Segment_Cells(img[1], NUC=img[0], cell_diameter=args.cell_diameter, GPU=args.gpu)

    # This line eliminates cells touching the edge of the tile, and keep only cells with one nucleus,
    # and labels the cell and nucleus using the same ID number
    # My default, the mask files will have two channels, pre-clean (index 0), and post-clean (index 1)
    print(f"Cleaning and labeling segmentation for well {n_well}, tile {n_tile}...")
    nucs, cells = isf.Segment.Label_and_Clean(nucs, cells)

    # Create output directories if they don't exist
    nucs_dir = os.path.join(args.output_dir, 'nucs', f'well_{n_well}')
    cells_dir = os.path.join(args.output_dir, 'cells', f'well_{n_well}')
    
    os.makedirs(nucs_dir, exist_ok=True)
    os.makedirs(cells_dir, exist_ok=True)

    # Segmentation masks will be saved in to dir tree segmented/magX/type/well_n/...
    save_name_nuc = os.path.join(nucs_dir, f'Seg_Nuc-Well_{n_well}_Tile_{n_tile}.npy')
    save_name_cell = os.path.join(cells_dir, f'Seg_Cells-Well_{n_well}_Tile_{n_tile}.npy')
    
    np.save(save_name_nuc, nucs)
    np.save(save_name_cell, cells)

    print(f"Saved segmentation results to {save_name_nuc} and {save_name_cell}")

    # Generate and save visualization if image processing is successful
    try:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img[0])
        ax[0].set_title('DAPI Channel')
        ax[0].axis('off')
        
        # Show cell outlines on second channel
        ax[1].imshow(img[1])
        ax[1].set_title('Cell Outlines')
        
        # Plot nuclei outlines
        from cellpose import utils
        outlines_nuc = utils.masks_to_outlines(nucs[1] if nucs.ndim > 2 else nucs)
        outX_nuc, outY_nuc = np.nonzero(outlines_nuc)
        ax[1].scatter(outY_nuc, outX_nuc, s=0.1, c='cyan')
        
        # Plot cell outlines
        outlines_cells = utils.masks_to_outlines(cells[1] if cells.ndim > 2 else cells)
        outX_cells, outY_cells = np.nonzero(outlines_cells)
        ax[1].scatter(outY_cells, outX_cells, s=0.1, c='lawngreen')
        
        ax[1].axis('off')
        
        # Save visualization
        viz_dir = os.path.join(args.output_dir, 'visualization', f'well_{n_well}')
        os.makedirs(viz_dir, exist_ok=True)
        viz_path = os.path.join(viz_dir, f'Seg_Viz-Well_{n_well}_Tile_{n_tile}.png')
        plt.tight_layout()
        plt.savefig(viz_path, dpi=150)
        plt.close()
        print(f"Saved visualization to {viz_path}")
    except Exception as e:
        print(f"Warning: Could not save visualization: {str(e)}")

    # Save summary as JSON
    try:
        import json
        
        # Count number of nuclei and cells
        num_nuclei = len(np.unique(nucs[1] if nucs.ndim > 2 else nucs)) - 1  # Subtract 1 for background
        num_cells = len(np.unique(cells[1] if cells.ndim > 2 else cells)) - 1  # Subtract 1 for background
        
        summary = {
            "well": n_well,
            "tile": n_tile,
            "num_nuclei": int(num_nuclei),
            "num_cells": int(num_cells),
            "nuc_diameter": args.nuc_diameter,
            "cell_diameter": args.cell_diameter,
            "processing_time": str(datetime.now() - t_start)
        }
        
        summary_dir = os.path.join(args.output_dir, 'summary', f'well_{n_well}')
        os.makedirs(summary_dir, exist_ok=True)
        summary_path = os.path.join(summary_dir, f'Seg_Summary-Well_{n_well}_Tile_{n_tile}.json')
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Saved summary to {summary_path}")
    except Exception as e:
        print(f"Warning: Could not save summary: {str(e)}")

    t_end = datetime.now()
    print(f'Well: {n_well} Tile: {n_tile} Time Start: {t_start} Time End: {t_end} Duration: {t_end - t_start}')
    print(f'Segmented {num_nuclei} nuclei and {num_cells} cells')