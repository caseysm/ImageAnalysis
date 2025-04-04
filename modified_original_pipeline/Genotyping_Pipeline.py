import sys, os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import glob
import re
import json
from pathlib import Path

# Add parent directory to import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from original pipeline
from original_pipeline import In_Situ_Functions as isf
import warnings
warnings.filterwarnings('ignore')


def find_segmentation_files(segmentation_dir, well, tile):
    """Find nuclei and cell mask files for given well and tile."""
    # Check for files in original format
    orig_nuc_path = os.path.join(segmentation_dir, 'nucs', f'well_{well}', f'Seg_Nuc-Well_{well}_Tile_{tile}.npy')
    orig_cell_path = os.path.join(segmentation_dir, 'cells', f'well_{well}', f'Seg_Cells-Well_{well}_Tile_{tile}.npy')
    
    if os.path.exists(orig_nuc_path) and os.path.exists(orig_cell_path):
        return orig_nuc_path, orig_cell_path
    
    # Check for files in new format
    # Search for *nuclei_mask.npy and *cell_mask.npy
    well_dir = os.path.join(segmentation_dir, f'Well{well}')
    if os.path.exists(well_dir):
        # Try to find a tile match
        nuc_files = glob.glob(os.path.join(well_dir, f"*Tile_{tile}*nuclei_mask.npy"))
        cell_files = glob.glob(os.path.join(well_dir, f"*Tile_{tile}*cell_mask.npy"))
        
        if nuc_files and cell_files:
            return nuc_files[0], cell_files[0]
            
        # If not found, try with sequence number
        nuc_files = glob.glob(os.path.join(well_dir, f"*Seq{tile:04d}*nuclei_mask.npy"))
        cell_files = glob.glob(os.path.join(well_dir, f"*Seq{tile:04d}*cell_mask.npy"))
        
        if nuc_files and cell_files:
            return nuc_files[0], cell_files[0]
    
    # If still not found, check for simplified filenames
    nuc_file = os.path.join(well_dir, "nuclei_mask.npy")
    cell_file = os.path.join(well_dir, "cell_mask.npy")
    
    if os.path.exists(nuc_file) and os.path.exists(cell_file):
        return nuc_file, cell_file
        
    raise FileNotFoundError(f"Could not find segmentation files for well {well}, tile {tile}")


def load_segmentation_masks(nuc_path, cell_path):
    """Load nuclei and cell masks from files."""
    # Handle different formats
    nucs = np.load(nuc_path)
    cells = np.load(cell_path)
    
    # Check if the format is (image, mask) tuple or just mask
    if isinstance(nucs, tuple) or (isinstance(nucs, np.ndarray) and nucs.ndim > 2):
        nucs = nucs[1]  # Extract mask from tuple
    
    if isinstance(cells, tuple) or (isinstance(cells, np.ndarray) and cells.ndim > 2):
        cells = cells[1]  # Extract mask from tuple
        
    return nucs, cells


def extract_well_tile(filename):
    """Extract well and tile information from filename."""
    # Try Well1_Point1_... format
    well_match = re.search(r'Well(\d+)', filename)
    if well_match:
        well = int(well_match.group(1))
    else:
        well = 1
        
    # Try to get tile from Point1 or Seq0123
    point_match = re.search(r'Point(\d+)', filename)
    if point_match:
        tile = int(point_match.group(1))
    else:
        seq_match = re.search(r'Seq(\d+)', filename)
        if seq_match:
            tile = int(seq_match.group(1))
        else:
            tile = 1
            
    return well, tile


def make_cycle_dirs(cycle_dirs, base_dir):
    """Create cycle directories structure."""
    results = []
    
    # Check if cycle_dirs is a single cycle directory
    if os.path.isdir(cycle_dirs):
        results = [cycle_dirs]
    else:
        # Check if it's a pattern like 'cycle_*'
        if isinstance(cycle_dirs, str) and '*' in cycle_dirs:
            cycle_pattern = os.path.join(base_dir, cycle_dirs)
            results = sorted(glob.glob(cycle_pattern))
        # Check if it's a list of directories
        elif isinstance(cycle_dirs, list):
            results = cycle_dirs
            
    if not results:
        raise ValueError(f"Could not find cycle directories using: {cycle_dirs}")
        
    return results


if __name__ == '__main__':
    t_start = datetime.now()

    parser = argparse.ArgumentParser(description="Genotype cells using barcode information")
    parser.add_argument('input_file', help='Path to input ND2 file or cycle directory')
    parser.add_argument('--segmentation-dir', required=True, help='Directory containing segmentation results')
    parser.add_argument('--barcode-library', required=True, help='Path to barcode library CSV file')
    parser.add_argument('--output-dir', default='genotyping_results', help='Output directory')
    parser.add_argument('--peak-threshold', type=float, default=200, help='Threshold for peak calling')
    parser.add_argument('--quality-threshold', type=float, default=0.3, help='Quality threshold for base calling')
    parser.add_argument('--cycles-dir', help='Directory containing genotyping cycles')
    parser.add_argument('--well', type=int, help='Well number (if not specified in filename)')
    parser.add_argument('--tile', type=int, help='Tile number (if not specified in filename)')
    args = parser.parse_args()

    try:
        # Extract well and tile info from filename or CLI args
        input_path = args.input_file
        filename = os.path.basename(input_path)
        
        if args.well is not None:
            well = args.well
        else:
            well, _ = extract_well_tile(filename)
            
        if args.tile is not None:
            tile = args.tile
        else:
            _, tile = extract_well_tile(filename)
            
        print(f"Processing {filename} - Well: {well}, Tile: {tile}")
        
        # Set up output directory
        output_dir = os.path.join(args.output_dir, f"well_{well}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Load barcode library
        df_lib = pd.read_csv(args.barcode_library)
        
        # Filter library by well group if needed (like in original)
        if well == 1:
            if 'group' in df_lib.columns:
                df_lib = df_lib[df_lib['group'] == 1]
        elif well == 2 or well == 3:
            if 'group' in df_lib.columns:
                df_lib = df_lib[df_lib['group'] == 3]
        elif well >= 4:
            if 'group' in df_lib.columns:
                df_lib = df_lib[df_lib['group'] == 2]
                
        # Determine cycle directories
        base_dir = os.path.dirname(input_path)
        if args.cycles_dir:
            cycle_dirs = make_cycle_dirs(args.cycles_dir, base_dir)
        else:
            # Try to find cycle directories in standard locations
            genotyping_dir = os.path.join(os.path.dirname(base_dir), 'genotyping')
            if os.path.exists(genotyping_dir):
                cycle_dirs = make_cycle_dirs('cycle_*', genotyping_dir)
            else:
                # If input_file is already a cycle directory
                parent_dir = os.path.dirname(input_path)
                if os.path.basename(parent_dir).startswith('cycle_'):
                    cycle_dirs = [parent_dir]
                else:
                    # Just use the directory containing the input file
                    cycle_dirs = [os.path.dirname(input_path)]
                    
        # Process cycle data for the specified tile and well
        print(f"Using cycle directories: {cycle_dirs}")
        
        # Try to assemble data from ND2 files
        try:
            # Modify this to use the provided cycle directories
            # Our custom implementation for this test
            data = []
            for cycle_dir in cycle_dirs:
                cycle_files = glob.glob(os.path.join(cycle_dir, f"*Well{well}*Point{tile}*.nd2"))
                if not cycle_files:
                    cycle_files = glob.glob(os.path.join(cycle_dir, f"*Well{well}*Seq{tile:04d}*.nd2"))
                
                if cycle_files:
                    cycle_file = cycle_files[0]
                    print(f"Processing cycle file: {cycle_file}")
                    
                    # Use isf module to load the data
                    from nd2reader import ND2Reader
                    with ND2Reader(cycle_file) as images:
                        channels = images.sizes['c']
                        cycle_data = np.zeros((channels, images.sizes['y'], images.sizes['x']))
                        for c in range(channels):
                            images.default_coords['c'] = c
                            cycle_data[c] = images[0]
                        data.append(cycle_data)
                else:
                    print(f"Warning: No files found in {cycle_dir} for Well{well}, Tile{tile}")
            
            if not data:
                raise ValueError("No cycle data found. Check cycle directories and file patterns.")
                
            data = np.array(data)  # Convert to numpy array with shape (cycles, channels, height, width)
            print(f"Assembled data shape: {data.shape}")
            
        except Exception as e:
            print(f"Error assembling cycle data: {str(e)}")
            print("Falling back to In_Situ_Functions implementation...")
            
            # Try the original implementation
            # Note: This may require the exact same directory structure as original
            data = isf.InSitu.Assemble_Data_From_ND2(tile, well, 'genotyping')
            
        # Find peaks in the data
        print("Finding peaks...")
        maxed, peaks, _ = isf.InSitu.Find_Peaks(data, verbose=False)
        
        # Load segmentation masks
        print("Loading segmentation masks...")
        try:
            nuc_path, cell_path = find_segmentation_files(args.segmentation_dir, well, tile)
            print(f"Found segmentation files: {nuc_path}, {cell_path}")
            nucs, cells = load_segmentation_masks(nuc_path, cell_path)
        except FileNotFoundError as e:
            print(f"Error: {str(e)}")
            sys.exit(1)
            
        # Count valid cells and nuclei
        num_cells = len(np.unique(cells)) - 1  # Subtract background
        num_nuclei = len(np.unique(nucs)) - 1  # Subtract background
        print(f"Loaded segmentation masks: {num_nuclei} nuclei, {num_cells} cells")
        
        # Call bases from peaks using given threshold
        print(f"Calling bases with threshold {args.peak_threshold}...")
        df_reads, _ = isf.InSitu.Call_Bases(cells, maxed, peaks, args.peak_threshold)
        print(f"Called {len(df_reads)} reads")
        
        # Assign ambiguity codes
        print(f"Assigning ambiguity with quality threshold {args.quality_threshold}...")
        df_reads_amb = isf.InSitu.Assign_Simple_Ambiguity(df_reads, lim=args.quality_threshold)
        
        # Choose best barcode for each cell
        print("Assigning genotypes to cells...")
        df_cell_genotype = isf.Lookup.Choose_Barcodes(df_reads_amb, df_lib, nucs, cells, verbose=True)
        print(f"Assigned genotypes to {len(df_cell_genotype)} cells")
        
        # Save results in original format
        print(f"Saving results to {output_dir}...")
        isf.Save(output_dir, tile, well, df_reads=df_reads, df_reads_amb=df_reads_amb, df_cell_genotype=df_cell_genotype)
        
        # Also save in the new format if the output_dir is different from default
        if args.output_dir != 'genotyping_results':
            # Save to the well directory in new format
            well_dir = os.path.join(args.output_dir, f"Well{well}")
            os.makedirs(well_dir, exist_ok=True)
            
            # Save genotypes CSV
            genotypes_file = os.path.join(well_dir, f"Well{well}_genotypes.csv")
            df_cell_genotype.to_csv(genotypes_file, index=False)
            
            # Create summary
            summary = {
                'well': f"Well{well}",
                'tile': tile,
                'total_cells': num_cells,
                'assigned_cells': len(df_cell_genotype),
                'processing_time': str(datetime.now() - t_start)
            }
            
            with open(os.path.join(well_dir, "summary.json"), 'w') as f:
                json.dump(summary, f, indent=2)
                
        t_end = datetime.now()
        print(f'Time: {t_end-t_start} Start: {t_start} Finish: {t_end}')
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)