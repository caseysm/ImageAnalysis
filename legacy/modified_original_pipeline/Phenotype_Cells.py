import sys, os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import glob
import json
import re
from pathlib import Path
import nd2reader

# Add parent directory to import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from original pipeline
from original_pipeline import In_Situ_Functions as isf
import warnings
warnings.filterwarnings('ignore')


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


def load_genotyping_results(genotyping_dir, well):
    """Load genotyping results for a well."""
    # Check for files in original format
    orig_file = os.path.join(genotyping_dir, f'well_{well}', f'genotypes-Well_{well}.csv')
    if os.path.exists(orig_file):
        return pd.read_csv(orig_file)
    
    # Check for files in new format
    new_file = os.path.join(genotyping_dir, f'Well{well}', f'Well{well}_genotypes.csv')
    if os.path.exists(new_file):
        return pd.read_csv(new_file)
        
    # Look for any *_genotypes.csv file
    well_dir = os.path.join(genotyping_dir, f'Well{well}')
    if os.path.exists(well_dir):
        geno_files = glob.glob(os.path.join(well_dir, '*_genotypes.csv'))
        if geno_files:
            return pd.read_csv(geno_files[0])
            
    return None


def measure_cell_properties(img, channels, cell_mask, nuc_mask, cell_id):
    """Measure properties for a specific cell."""
    # Get cell and nucleus masks for this cell
    cell_region = (cell_mask == cell_id)
    nuc_region = (nuc_mask == cell_id)
    
    # Skip if cell is empty
    if not np.any(cell_region):
        return None
        
    # Basic properties
    from skimage.measure import regionprops
    cell_props = regionprops(cell_region.astype(int))[0]
    
    # Initialize properties dict
    properties = {
        'cell_id': cell_id,
        'area': cell_props.area,
        'perimeter': cell_props.perimeter,
        'eccentricity': cell_props.eccentricity,
        'solidity': cell_props.solidity,
        'major_axis_length': cell_props.major_axis_length,
        'minor_axis_length': cell_props.minor_axis_length,
        'centroid_y': cell_props.centroid[0],
        'centroid_x': cell_props.centroid[1]
    }
    
    # Circularity
    if cell_props.perimeter > 0:
        properties['circularity'] = (4 * np.pi * cell_props.area) / (cell_props.perimeter ** 2)
    else:
        properties['circularity'] = 0
        
    # Nucleus properties if present
    if np.any(nuc_region):
        nuc_props = regionprops(nuc_region.astype(int))[0]
        properties['nucleus_area'] = nuc_props.area
        properties['nucleus_perimeter'] = nuc_props.perimeter
        properties['nucleus_to_cell_ratio'] = nuc_props.area / cell_props.area if cell_props.area > 0 else 0
    
    # Intensity measures for each channel
    for c, channel_name in enumerate(channels):
        if c < img.shape[0]:  # Make sure channel exists
            channel_img = img[c]
            
            # Cell intensity measurements
            cell_pixels = channel_img[cell_region]
            if len(cell_pixels) > 0:
                properties[f'{channel_name}_mean_intensity'] = np.mean(cell_pixels)
                properties[f'{channel_name}_max_intensity'] = np.max(cell_pixels)
                properties[f'{channel_name}_std_intensity'] = np.std(cell_pixels)
                properties[f'{channel_name}_total_intensity'] = np.sum(cell_pixels)
            else:
                properties[f'{channel_name}_mean_intensity'] = 0
                properties[f'{channel_name}_max_intensity'] = 0
                properties[f'{channel_name}_std_intensity'] = 0
                properties[f'{channel_name}_total_intensity'] = 0
                
            # Nucleus intensity measurements if present
            if np.any(nuc_region):
                nuc_pixels = channel_img[nuc_region]
                if len(nuc_pixels) > 0:
                    properties[f'{channel_name}_nucleus_mean_intensity'] = np.mean(nuc_pixels)
                    properties[f'{channel_name}_nucleus_max_intensity'] = np.max(nuc_pixels)
                else:
                    properties[f'{channel_name}_nucleus_mean_intensity'] = 0
                    properties[f'{channel_name}_nucleus_max_intensity'] = 0
    
    return properties


if __name__ == '__main__':
    t_start = datetime.now()

    parser = argparse.ArgumentParser(description="Extract phenotypic features from segmented cells")
    parser.add_argument('input_file', help='Path to input ND2 file with images')
    parser.add_argument('--segmentation-dir', required=True, help='Directory containing segmentation results')
    parser.add_argument('--channels', nargs='+', required=True, help='Channel names in order (e.g., DAPI GFP RFP)')
    parser.add_argument('--genotyping-dir', help='Directory containing genotyping results (optional)')
    parser.add_argument('--output-dir', default='phenotyping_results', help='Output directory')
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
        output_dir = os.path.join(args.output_dir, f"Well{well}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Load ND2 file
        print("Loading ND2 file...")
        with nd2reader.ND2Reader(input_path) as images:
            # Get dimensions
            channels = images.sizes.get('c', 1)
            if channels < len(args.channels):
                print(f"Warning: File has {channels} channels but {len(args.channels)} channel names provided")
                
            # Load all channels
            img_data = np.zeros((channels, images.sizes['y'], images.sizes['x']))
            for c in range(channels):
                images.default_coords['c'] = c
                img_data[c] = images[0]
                
        # Load segmentation masks
        print("Loading segmentation masks...")
        try:
            nuc_path, cell_path = find_segmentation_files(args.segmentation_dir, well, tile)
            print(f"Found segmentation files: {nuc_path}, {cell_path}")
            nuc_mask, cell_mask = load_segmentation_masks(nuc_path, cell_path)
        except FileNotFoundError as e:
            print(f"Error: {str(e)}")
            sys.exit(1)
            
        # Count valid cells and nuclei
        unique_cells = np.unique(cell_mask)
        unique_cells = unique_cells[unique_cells > 0]  # Remove background
        num_cells = len(unique_cells)
        print(f"Loaded segmentation masks: {num_cells} cells")
        
        # Load genotyping results if available
        genotypes_df = None
        if args.genotyping_dir:
            print("Loading genotyping results...")
            genotypes_df = load_genotyping_results(args.genotyping_dir, well)
            if genotypes_df is not None:
                print(f"Loaded genotypes for {len(genotypes_df)} cells")
            else:
                print("No genotyping results found")
        
        # Measure cell properties
        print("Measuring cell properties...")
        all_properties = []
        
        for cell_id in unique_cells:
            props = measure_cell_properties(img_data, args.channels, cell_mask, nuc_mask, cell_id)
            if props is not None:
                # Add genotype information if available
                if genotypes_df is not None:
                    cell_genotype = genotypes_df[genotypes_df['cell_id'] == cell_id]
                    if not cell_genotype.empty:
                        if 'barcode' in cell_genotype.columns:
                            props['barcode'] = cell_genotype.iloc[0]['barcode']
                        if 'sgRNA' in cell_genotype.columns:
                            props['sgRNA'] = cell_genotype.iloc[0]['sgRNA']
                            
                all_properties.append(props)
                
        # Create DataFrame from properties
        print(f"Processed {len(all_properties)} cells")
        if all_properties:
            df = pd.DataFrame(all_properties)
            
            # Save output
            csv_path = os.path.join(output_dir, f"Well{well}_phenotypes.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved phenotype data to {csv_path}")
            
            # Create summary
            summary = {
                'well': f"Well{well}",
                'tile': tile,
                'total_cells': num_cells,
                'phenotyped_cells': len(all_properties),
                'channels': args.channels,
                'processing_time': str(datetime.now() - t_start)
            }
            
            with open(os.path.join(output_dir, "summary.json"), 'w') as f:
                json.dump(summary, f, indent=2)
        else:
            print("No valid cells processed")
            
        t_end = datetime.now()
        print(f'Processing time: {t_end-t_start}')
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)