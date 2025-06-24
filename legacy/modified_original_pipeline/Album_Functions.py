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
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

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


def load_phenotype_data(phenotyping_dir, well):
    """Load phenotyping data for a well."""
    # Check for files in various formats
    
    # New format
    phenotype_file = os.path.join(phenotyping_dir, f"Well{well}", f"Well{well}_phenotypes.csv")
    if os.path.exists(phenotype_file):
        return pd.read_csv(phenotype_file)
        
    # Look for any *_phenotypes.csv file
    well_dir = os.path.join(phenotyping_dir, f"Well{well}")
    if os.path.exists(well_dir):
        pheno_files = glob.glob(os.path.join(well_dir, '*_phenotypes.csv'))
        if pheno_files:
            return pd.read_csv(pheno_files[0])
            
    # Original format (if it exists)
    orig_file = os.path.join(phenotyping_dir, f"well_{well}", f"phenotypes-Well_{well}.csv")
    if os.path.exists(orig_file):
        return pd.read_csv(orig_file)
        
    raise FileNotFoundError(f"Could not find phenotype data for well {well}")


def find_segmentation_files(segmentation_dir, well, tile=None):
    """Find nuclei and cell mask files for given well."""
    # First, try to find individual tile-specific files
    if tile is not None:
        # Check for files in original format
        orig_nuc_path = os.path.join(segmentation_dir, 'nucs', f'well_{well}', f'Seg_Nuc-Well_{well}_Tile_{tile}.npy')
        orig_cell_path = os.path.join(segmentation_dir, 'cells', f'well_{well}', f'Seg_Cells-Well_{well}_Tile_{tile}.npy')
        
        if os.path.exists(orig_nuc_path) and os.path.exists(orig_cell_path):
            return orig_nuc_path, orig_cell_path
        
        # Check for files in new format with specific tile/sequence
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
    
    # If tile-specific files not found or tile not specified, look for well-level files
    # Check for simplified filenames
    well_dir = os.path.join(segmentation_dir, f'Well{well}')
    if os.path.exists(well_dir):
        nuc_file = os.path.join(well_dir, "nuclei_mask.npy")
        cell_file = os.path.join(well_dir, "cell_mask.npy")
        
        if os.path.exists(nuc_file) and os.path.exists(cell_file):
            return nuc_file, cell_file
            
        # Try to find any nuclei/cell mask files
        nuc_files = glob.glob(os.path.join(well_dir, '*nuclei_mask.npy'))
        cell_files = glob.glob(os.path.join(well_dir, '*cell_mask.npy'))
        
        if nuc_files and cell_files:
            return nuc_files[0], cell_files[0]
        
    raise FileNotFoundError(f"Could not find segmentation files for well {well}")


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


def crop_cell(image, cell_mask, cell_id, padding=5):
    """Crop a single cell from the image."""
    # Get cell mask for this specific cell
    cell_region = (cell_mask == cell_id)
    
    # Find bounding box
    if not np.any(cell_region):
        return None, None, None, None, None
        
    rows, cols = np.where(cell_region)
    y_min, y_max = np.min(rows), np.max(rows)
    x_min, x_max = np.min(cols), np.max(cols)
    
    # Add padding
    y_min = max(0, y_min - padding)
    y_max = min(cell_mask.shape[0] - 1, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(cell_mask.shape[1] - 1, x_max + padding)
    
    # Crop image and mask
    if image is not None:
        if len(image.shape) == 3:  # Multi-channel image
            # Check if channels are first or last dimension
            if image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
                # Channels first (c, h, w)
                cropped_img = image[:, y_min:y_max+1, x_min:x_max+1]
            else:
                # Channels last (h, w, c)
                cropped_img = image[y_min:y_max+1, x_min:x_max+1, :]
        else:
            # Single channel
            cropped_img = image[y_min:y_max+1, x_min:x_max+1]
    else:
        cropped_img = None
        
    cropped_mask = cell_region[y_min:y_max+1, x_min:x_max+1]
    
    return cropped_img, cropped_mask, y_min, y_max, x_min, x_max


def resize_to_square(img, size):
    """Resize a cell image to a square of the specified size."""
    if img is None:
        return None
        
    # Get current dimensions
    if len(img.shape) == 3:
        # Check if channels are first or last dimension
        if img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
            # Channels first (c, h, w)
            channels, height, width = img.shape
            is_channels_first = True
        else:
            # Channels last (h, w, c)
            height, width, channels = img.shape
            is_channels_first = False
    else:
        height, width = img.shape
        channels = 1
        is_channels_first = False
        
    # Determine the maximum dimension
    max_dim = max(height, width)
    
    # Create a square canvas
    if is_channels_first:
        square_img = np.zeros((channels, max_dim, max_dim), dtype=img.dtype)
        # Center the original image
        y_offset = (max_dim - height) // 2
        x_offset = (max_dim - width) // 2
        square_img[:, y_offset:y_offset+height, x_offset:x_offset+width] = img
    elif channels > 1:
        square_img = np.zeros((max_dim, max_dim, channels), dtype=img.dtype)
        y_offset = (max_dim - height) // 2
        x_offset = (max_dim - width) // 2
        square_img[y_offset:y_offset+height, x_offset:x_offset+width, :] = img
    else:
        square_img = np.zeros((max_dim, max_dim), dtype=img.dtype)
        y_offset = (max_dim - height) // 2
        x_offset = (max_dim - width) // 2
        square_img[y_offset:y_offset+height, x_offset:x_offset+width] = img
        
    # Resize to target size
    from skimage.transform import resize
    
    if is_channels_first:
        resized_img = np.zeros((channels, size, size), dtype=np.float32)
        for c in range(channels):
            resized_img[c] = resize(square_img[c], (size, size), preserve_range=True)
    elif channels > 1:
        resized_img = resize(square_img, (size, size, channels), preserve_range=True)
    else:
        resized_img = resize(square_img, (size, size), preserve_range=True)
        
    return resized_img


def normalize_image(img, percentile=99.5):
    """Normalize image to [0, 1] range."""
    if img is None or np.all(img == 0):
        return img
        
    # Handle different image formats
    if len(img.shape) == 3:
        # Check if channels are first or last dimension
        if img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
            # Channels first (c, h, w)
            channels = img.shape[0]
            normalized = np.zeros_like(img, dtype=np.float32)
            for c in range(channels):
                channel = img[c]
                if np.any(channel > 0):
                    min_val = np.min(channel)
                    max_val = np.percentile(channel, percentile)
                    if max_val > min_val:
                        normalized[c] = np.clip((channel - min_val) / (max_val - min_val), 0, 1)
        else:
            # Channels last (h, w, c)
            channels = img.shape[2]
            normalized = np.zeros_like(img, dtype=np.float32)
            for c in range(channels):
                channel = img[:, :, c]
                if np.any(channel > 0):
                    min_val = np.min(channel)
                    max_val = np.percentile(channel, percentile)
                    if max_val > min_val:
                        normalized[:, :, c] = np.clip((channel - min_val) / (max_val - min_val), 0, 1)
    else:
        # Single channel
        if np.any(img > 0):
            min_val = np.min(img)
            max_val = np.percentile(img, percentile)
            if max_val > min_val:
                normalized = np.clip((img - min_val) / (max_val - min_val), 0, 1)
            else:
                normalized = np.zeros_like(img, dtype=np.float32)
        else:
            normalized = np.zeros_like(img, dtype=np.float32)
            
    return normalized


def create_album(image_data, phenotypes_df, cell_mask, channels, cell_size=128, grid_size=(5, 10), sort_by=None, group_by=None, group_value=None):
    """Create an album of cells arranged in a grid."""
    # Filter phenotypes if grouping is specified
    if group_by is not None and group_by in phenotypes_df.columns:
        if group_value is not None:
            phenotypes_df = phenotypes_df[phenotypes_df[group_by] == group_value]
            
    # Sort if specified
    if sort_by is not None and sort_by in phenotypes_df.columns:
        phenotypes_df = phenotypes_df.sort_values(by=sort_by, ascending=False)
        
    # Limit to grid size
    rows, cols = grid_size
    max_cells = rows * cols
    cell_ids = phenotypes_df['cell_id'].tolist()[:max_cells]
    
    # Prepare the album canvas
    album_height = rows * cell_size
    album_width = cols * cell_size
    
    # Check image_data format - we'll assume channels last format for the album
    if image_data is not None:
        if len(image_data.shape) == 3:
            if image_data.shape[0] < image_data.shape[1] and image_data.shape[0] < image_data.shape[2]:
                # Channels first (c, h, w)
                n_channels = image_data.shape[0]
                channels_first = True
            else:
                # Channels last (h, w, c)
                n_channels = image_data.shape[2]
                channels_first = False
        else:
            n_channels = 1
            channels_first = False
            
        if n_channels == 1:
            # Single channel to RGB
            album = np.zeros((album_height, album_width, 3), dtype=np.float32)
        elif n_channels == 2:
            # 2 channels to RGB
            album = np.zeros((album_height, album_width, 3), dtype=np.float32)
        elif n_channels == 3:
            # 3 channels to RGB
            album = np.zeros((album_height, album_width, 3), dtype=np.float32)
        else:
            # Multi-channel to RGB
            album = np.zeros((album_height, album_width, 3), dtype=np.float32)
    else:
        # If no image data, create RGB album
        n_channels = len(channels) if channels else 3
        channels_first = False
        album = np.zeros((album_height, album_width, 3), dtype=np.float32)
    
    # Process each cell
    cell_thumbnails = {}
    for idx, cell_id in enumerate(cell_ids):
        if idx >= max_cells:
            break
            
        # Calculate position in the grid
        row = idx // cols
        col = idx % cols
        
        # Create cell thumbnail
        if image_data is not None:
            # Crop cell from main image
            cropped_img, cropped_mask, y1, y2, x1, x2 = crop_cell(image_data, cell_mask, cell_id)
            
            if cropped_img is not None:
                # Resize to square
                cell_thumb = resize_to_square(cropped_img, cell_size)
                
                # Normalize thumbnail
                cell_thumb = normalize_image(cell_thumb)
                
                # Handle channel conversion to RGB
                if channels_first:
                    if n_channels == 1:
                        # Single channel to grayscale
                        thumb_rgb = np.stack([cell_thumb[0]]*3, axis=2)
                    elif n_channels == 2:
                        # 2 channels: R=ch0, G=ch1, B=0
                        thumb_rgb = np.zeros((cell_size, cell_size, 3))
                        thumb_rgb[:, :, 0] = cell_thumb[0]  # R
                        thumb_rgb[:, :, 1] = cell_thumb[1]  # G
                    elif n_channels == 3:
                        # 3 channels: R=ch0, G=ch1, B=ch2
                        thumb_rgb = np.stack([
                            cell_thumb[0],  # R
                            cell_thumb[1],  # G
                            cell_thumb[2]   # B
                        ], axis=2)
                    else:
                        # More than 3 channels: use first 3
                        thumb_rgb = np.stack([
                            cell_thumb[0],  # R
                            cell_thumb[1],  # G
                            cell_thumb[2]   # B
                        ], axis=2)
                else:
                    if n_channels == 1:
                        # Single channel to grayscale
                        thumb_rgb = np.stack([cell_thumb]*3, axis=2)
                    elif n_channels == 3:
                        # Already RGB
                        thumb_rgb = cell_thumb
                    else:
                        # Use as is, but ensure RGB format
                        thumb_rgb = cell_thumb[:, :, :3] if cell_thumb.shape[2] >= 3 else np.pad(
                            cell_thumb, ((0, 0), (0, 0), (0, 3 - cell_thumb.shape[2])),
                            mode='constant'
                        )
            else:
                # Create empty thumbnail
                thumb_rgb = np.zeros((cell_size, cell_size, 3))
        else:
            # Create placeholder thumbnail using phenotype data
            thumb_rgb = np.zeros((cell_size, cell_size, 3))
            
            # Use phenotype data to create a visual representation
            cell_row = phenotypes_df[phenotypes_df['cell_id'] == cell_id]
            if not cell_row.empty:
                # Draw a circle with size proportional to cell area
                if 'area' in cell_row.columns:
                    area = float(cell_row['area'].iloc[0])
                    radius = int(np.sqrt(area / np.pi) * 0.5)  # Scale to fit thumbnail
                    radius = min(radius, cell_size // 2 - 5)  # Limit radius
                else:
                    radius = cell_size // 4
                
                # Add color based on channels
                center = cell_size // 2
                y, x = np.ogrid[:cell_size, :cell_size]
                mask = ((x - center)**2 + (y - center)**2) < radius**2
                
                # Color by channel intensities if available
                for i, channel in enumerate(channels[:3]):  # Limit to RGB channels
                    intensity_col = f"{channel}_intensity"
                    if intensity_col in cell_row.columns:
                        intensity = float(cell_row[intensity_col].iloc[0])
                        thumb_rgb[mask, i] = intensity
        
        # Add to album
        y_start = row * cell_size
        x_start = col * cell_size
        album[y_start:y_start+cell_size, x_start:x_start+cell_size] = thumb_rgb
        
        # Save thumbnail
        cell_thumbnails[cell_id] = thumb_rgb
        
    return album, cell_thumbnails, cell_ids


if __name__ == '__main__':
    t_start = datetime.now()

    parser = argparse.ArgumentParser(description="Create cell albums from phenotyping data")
    parser.add_argument('input_file', nargs='?', help='Path to input ND2 file (optional)')
    parser.add_argument('--phenotyping-dir', required=True, help='Directory containing phenotyping results')
    parser.add_argument('--segmentation-dir', help='Directory containing segmentation results (optional)')
    parser.add_argument('--channels', nargs='+', help='Channel names (e.g., DAPI GFP RFP)')
    parser.add_argument('--output-dir', default='album_results', help='Output directory')
    parser.add_argument('--well', help='Well to process (e.g., Well1 or 1)')
    parser.add_argument('--cell-size', type=int, default=128, help='Size of cell thumbnails in pixels')
    parser.add_argument('--grid-rows', type=int, default=5, help='Number of rows in album grid')
    parser.add_argument('--grid-cols', type=int, default=10, help='Number of columns in album grid')
    parser.add_argument('--sort-by', help='Column to sort cells by')
    parser.add_argument('--group-by', help='Column to group cells by')
    parser.add_argument('--group-value', help='Value to filter group_by column')
    args = parser.parse_args()

    try:
        # Process well argument
        if args.well:
            if args.well.startswith('Well'):
                well_str = args.well
                well_num = int(args.well.replace('Well', ''))
            else:
                well_num = int(args.well)
                well_str = f"Well{well_num}"
        else:
            # Try to extract from input file if provided
            if args.input_file:
                filename = os.path.basename(args.input_file)
                well_num, _ = extract_well_tile(filename)
                well_str = f"Well{well_num}"
            else:
                # Look for any well in phenotyping directory
                pheno_dir = args.phenotyping_dir
                well_dirs = [d for d in os.listdir(pheno_dir) if os.path.isdir(os.path.join(pheno_dir, d)) and d.startswith('Well')]
                if well_dirs:
                    well_str = well_dirs[0]
                    well_num = int(well_str.replace('Well', ''))
                else:
                    raise ValueError("Could not determine well to process. Please specify with --well.")
                    
        print(f"Processing well: {well_str}")
        
        # Set up output directory
        output_dir = os.path.join(args.output_dir, well_str)
        os.makedirs(output_dir, exist_ok=True)
        
        # Load phenotype data
        print("Loading phenotype data...")
        try:
            phenotypes_df = load_phenotype_data(args.phenotyping_dir, well_num)
            print(f"Loaded phenotype data for {len(phenotypes_df)} cells")
        except FileNotFoundError as e:
            print(f"Error: {str(e)}")
            sys.exit(1)
            
        # Load image data if available
        image_data = None
        cell_mask = None
        
        if args.input_file:
            print("Loading image data...")
            try:
                with nd2reader.ND2Reader(args.input_file) as images:
                    # Get dimensions
                    channels = images.sizes.get('c', 1)
                    if args.channels and channels < len(args.channels):
                        print(f"Warning: File has {channels} channels but {len(args.channels)} channel names provided")
                        
                    # Load all channels
                    image_data = np.zeros((channels, images.sizes['y'], images.sizes['x']))
                    for c in range(channels):
                        images.default_coords['c'] = c
                        image_data[c] = images[0]
                        
                print(f"Loaded image data: {image_data.shape}")
            except Exception as e:
                print(f"Warning: Could not load image data: {str(e)}")
                image_data = None
                
        # Load segmentation masks if available
        if args.segmentation_dir:
            print("Loading segmentation masks...")
            try:
                # Try to find tile from input file
                tile = None
                if args.input_file:
                    _, tile = extract_well_tile(os.path.basename(args.input_file))
                    
                nuc_path, cell_path = find_segmentation_files(args.segmentation_dir, well_num, tile)
                _, cell_mask = load_segmentation_masks(nuc_path, cell_path)
                print(f"Loaded cell mask: {cell_mask.shape}, {len(np.unique(cell_mask))-1} cells")
            except FileNotFoundError as e:
                print(f"Warning: Could not load segmentation masks: {str(e)}")
                cell_mask = None
                
        # Set up grid size
        grid_size = (args.grid_rows, args.grid_cols)
        print(f"Using grid size: {grid_size[0]}x{grid_size[1]}")
        
        # Create album
        print("Creating cell album...")
        album, cell_thumbnails, cell_ids = create_album(
            image_data=image_data,
            phenotypes_df=phenotypes_df,
            cell_mask=cell_mask,
            channels=args.channels,
            cell_size=args.cell_size,
            grid_size=grid_size,
            sort_by=args.sort_by,
            group_by=args.group_by,
            group_value=args.group_value
        )
        
        # Save album image
        album_path = os.path.join(output_dir, f"{well_str}_album.npy")
        np.save(album_path, album)
        print(f"Saved album to: {album_path}")
        
        # Save individual cell thumbnails
        cells_dir = os.path.join(output_dir, "cells")
        os.makedirs(cells_dir, exist_ok=True)
        
        for cell_id, thumbnail in cell_thumbnails.items():
            thumb_path = os.path.join(cells_dir, f"cell_{cell_id}.npy")
            np.save(thumb_path, thumbnail)
            
        # Create metadata
        metadata = {
            'well': well_str,
            'grid_size': grid_size,
            'cells_in_album': len(cell_ids),
            'cell_ids': cell_ids,
            'channels': args.channels,
            'sort_by': args.sort_by,
            'group_by': args.group_by,
            'group_value': args.group_value,
            'cell_size': args.cell_size
        }
        
        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Also save a PNG version of the album
        print("Saving album as PNG...")
        plt.figure(figsize=(grid_size[1], grid_size[0]), dpi=args.cell_size)
        plt.imshow(album)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(output_dir, f"{well_str}_album.png"), bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"Successfully created album with {len(cell_ids)} cells")
        
        t_end = datetime.now()
        print(f'Processing time: {t_end-t_start}')
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)