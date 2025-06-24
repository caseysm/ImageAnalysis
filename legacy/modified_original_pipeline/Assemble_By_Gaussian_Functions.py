import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import scipy.ndimage as ndi
import scipy.signal as ss
import skimage.filters as sf
from skimage.morphology import disk
from sklearn import mixture
import itertools
import os, sys


def cluster_by_gaussian(_X, n_cluster=2, seed=None, verbose=False):

    # Decouple a 1D numpy array of values into gaussians peaks
    # _X: 1D numpy array of values
    # return: 1D numpy array of integer labels

    _X = np.expand_dims(_X, -1)

    if seed != 'None':
        np.random.seed(seed)

    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, n_cluster + 1)
    cv_types = ["spherical", "tied", "diag", "full"]
    for cv_type in cv_types:
        for n_components in n_components_range:
            gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type)
            gmm.fit(_X)
            bic.append(gmm.bic(_X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    color_iter = itertools.cycle(['black', 'purple', 'blue', 'green', 'gold', 'orange', 'red', 'pink', 'brown', 'grey'])
    # color_iter = itertools.cycle(['Greys_r', 'Purples_r', 'Blues_r', 'Greens_r', 'Oranges_r', 'Reds_r'])
    clf = best_gmm

    _Y = clf.predict(_X)

    if verbose:
        for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_, color_iter)):
            if not np.any(_Y == i):
                continue
            # plot_2D_hist(_X[_Y == i, 0], _X[_Y == i, 1], colormap=color)
            plt.scatter(_X[_Y == i, 0], _X[_Y == i, 1], s=0.2, color=color)

    return _Y


def Find_Intensity_Peaks(_z, bins=50, normalized_height=0.2, filter_width_coef=2, pad_left=True, pad_right=True,
                         peak_choice=-1, verbose=False):

    # Function to detect log mean single cell image intensity
    # Useful in extracting single cells of a certain sgRNA close the mean of the log normal distribution
    # Cells close to the high or low end of the distribution tend to exhibit anomalous protein localization pattern,
    # This function creates a historgram of the mean log intensities, and uses a smoothing gaussian filter to detect peaks.
    # based on the number of peaks, it then uses gaussian mixture to fit gaussian curves to the data.
    # This enable the function to handle single gaussian distributions, as well as multimodal distributions.
    # In the case of multimodal distributions, the function will focus on the highest intensity gaussian.
    # Once a gaussian is identified, are range will be caluclated of +/- one standard deviation to select cells from.

    # _z: 1D numpy array of the mean log single cell image intensities
    # bins: number of bins used to create the histrogram
    # normalized_height: when identifying peaks after hieght normalization, below this value peaks will be ignored.
    # designed to eliminate confusion between low frequency peaks and noise.
    # filter_width_coef: width of gaussian filter
    # pad_left: add zero value class at the lower end of the histogram, helps stabilize filter
    # pad_right: add zero value class at the higher end of the histogram, helps stabilize filter
    # peak_choice: In case of a multimodal distribution, and direct towards any peaks of interest.
    # By default, in a multimodal distribution, the the highest intensity gaussian peak will be chosen.
    # verbose: Plot the gaussian fitting process
    # return: range of +/- one std of log intensities to choose cells from, coordinates of identified peaks

    # Create histogram
    _h, _b = np.histogram(_z, bins=bins)
    _h = _h / _h.max()
    _db = _b[1] - _b[0]

    if pad_left:
        _b = np.concatenate(([_b[0] - _db], _b))
        _h = np.concatenate(([0], _h))

    if pad_right:
        _b = np.concatenate((_b, [_b[-1] + _db]))
        _h = np.concatenate((_h, [0]))

    _b = _b[:-1] / 2 + _b[1:] / 2

    # Perform gaussian smoothing
    _hs = ndi.gaussian_filter(_h, sigma=filter_width_coef * (_b.max() - _b.min()) / len(_b))

    # Identify peaks
    _peaks, _ = ss.find_peaks(_hs, height=normalized_height)

    # If no peaks found, consider the distribution as a single Gaussian
    if len(_peaks) == 0:
        _mean = np.mean(_z)
        _std = np.std(_z)
        
        if verbose:
            plt.figure(figsize=(8, 6))
            plt.hist(_z, bins=bins, alpha=0.5)
            plt.axvline(_mean, color='r', linestyle='--')
            plt.axvline(_mean - _std, color='g', linestyle=':')
            plt.axvline(_mean + _std, color='g', linestyle=':')
            plt.title('No Clear Peaks - Using Full Distribution')
            plt.show()
            
        return [_mean - _std, _mean + _std], []

    # If exactly one peak, treat as single Gaussian
    if len(_peaks) == 1:
        _gmm = mixture.GaussianMixture(n_components=1)
        _gmm.fit(_z.reshape(-1, 1))
        _mean = _gmm.means_[0, 0]
        _std = np.sqrt(_gmm.covariances_[0, 0, 0])
        
        if verbose:
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.plot(_b, _h, 'b-', label='Histogram')
            plt.plot(_b, _hs, 'r-', label='Smoothed')
            plt.plot(_b[_peaks], _hs[_peaks], 'go', label='Peak')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.hist(_z, bins=bins, alpha=0.5)
            plt.axvline(_mean, color='r', linestyle='--')
            plt.axvline(_mean - _std, color='g', linestyle=':')
            plt.axvline(_mean + _std, color='g', linestyle=':')
            plt.title('Single Peak Gaussian Fit')
            plt.show()
            
        return [_mean - _std, _mean + _std], _peaks

    # Multiple peaks - use Gaussian Mixture Model
    _gmm = mixture.GaussianMixture(n_components=len(_peaks))
    _gmm.fit(_z.reshape(-1, 1))
    
    # Choose appropriate peak
    if peak_choice == -1:  # Use highest intensity peak by default
        _components = [(m[0], np.sqrt(c[0, 0])) for m, c in zip(_gmm.means_, _gmm.covariances_)]
        _components_sorted = sorted(_components, key=lambda x: x[0])
        _mean, _std = _components_sorted[-1]  # Highest intensity peak
    else:
        # If specific peak is requested, sort by mean intensity and select
        _components = [(i, m[0], np.sqrt(c[0, 0])) for i, (m, c) in enumerate(zip(_gmm.means_, _gmm.covariances_))]
        _components_sorted = sorted(_components, key=lambda x: x[1])
        if peak_choice < len(_components_sorted):
            _, _mean, _std = _components_sorted[peak_choice]
        else:
            _, _mean, _std = _components_sorted[-1]  # Default to highest if choice is out of range
    
    if verbose:
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot(_b, _h, 'b-', label='Histogram')
        plt.plot(_b, _hs, 'r-', label='Smoothed')
        plt.plot(_b[_peaks], _hs[_peaks], 'go', label='Peaks')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.hist(_z, bins=bins, alpha=0.5)
        
        # Plot all mixture components
        x = np.linspace(min(_z), max(_z), 1000)
        for i, (weight, mean, cov) in enumerate(zip(_gmm.weights_, _gmm.means_, _gmm.covariances_)):
            pdf = weight * ss.norm.pdf(x, mean, np.sqrt(cov[0, 0]))
            plt.plot(x, pdf * len(_z) * (_b[1] - _b[0]), label=f'Component {i+1}')
            
        plt.axvline(_mean, color='r', linestyle='--', label='Selected Mean')
        plt.axvline(_mean - _std, color='g', linestyle=':', label='±1 Std Dev')
        plt.axvline(_mean + _std, color='g', linestyle=':')
        plt.legend()
        plt.title('Multiple Peak Gaussian Mixture Model')
        plt.show()
        
    return [_mean - _std, _mean + _std], _peaks


def Filter_Cells_By_Intensity(_df, _range, column_name='mean_intensity'):
    """
    Filter cells based on intensity range
    
    Parameters:
    -----------
    _df : pandas DataFrame
        DataFrame containing cell information
    _range : list
        [min, max] intensity range to filter cells
    column_name : str
        Column name in DataFrame containing intensity values
        
    Returns:
    --------
    pandas DataFrame
        Filtered DataFrame containing only cells within specified intensity range
    """
    return _df[(_df[column_name] >= _range[0]) & (_df[column_name] <= _range[1])]


def Extract_Representative_Cells(_df, n_cells=50, column_name='mean_intensity', peak_range=None, 
                                seed=None, verbose=False):
    """
    Extract representative cells from a dataset, optionally focusing on a specific intensity peak
    
    Parameters:
    -----------
    _df : pandas DataFrame
        DataFrame containing cell information
    n_cells : int
        Number of cells to extract
    column_name : str
        Column name in DataFrame containing intensity values
    peak_range : list or None
        Optional [min, max] intensity range to filter cells. If None, will be calculated.
    seed : int or None
        Random seed for reproducibility
    verbose : bool
        Whether to show plots
        
    Returns:
    --------
    pandas DataFrame
        DataFrame containing selected representative cells
    """
    if seed is not None:
        np.random.seed(seed)
    
    # If no peak range provided, calculate it
    if peak_range is None:
        intensity_values = _df[column_name].values
        peak_range, _ = Find_Intensity_Peaks(intensity_values, verbose=verbose)
    
    # Filter cells by intensity
    filtered_df = Filter_Cells_By_Intensity(_df, peak_range, column_name)
    
    if len(filtered_df) == 0:
        if verbose:
            print("No cells found in specified intensity range. Using full dataset.")
        filtered_df = _df
    
    # If we have fewer cells than requested, return all available
    if len(filtered_df) <= n_cells:
        if verbose:
            print(f"Only {len(filtered_df)} cells available in intensity range. Returning all.")
        return filtered_df
    
    # Otherwise, randomly sample n_cells
    indices = np.random.choice(len(filtered_df), n_cells, replace=False)
    return filtered_df.iloc[indices].reset_index(drop=True)


def Apply_Image_Filters(image, filters=None, verbose=False):
    """
    Apply a sequence of filters to an image
    
    Parameters:
    -----------
    image : numpy array
        Input image
    filters : list of tuples
        List of (filter_name, params) tuples defining filters to apply
        Supported filters: 'gaussian', 'median', 'threshold_otsu', 'threshold_local'
    verbose : bool
        Whether to show intermediate results
        
    Returns:
    --------
    numpy array
        Filtered image
    """
    result = image.copy()
    
    if filters is None:
        return result
        
    if verbose:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, len(filters)+1, 1)
        plt.imshow(result)
        plt.title('Original')
        plt.axis('off')
    
    for i, (filter_name, params) in enumerate(filters):
        if filter_name == 'gaussian':
            sigma = params.get('sigma', 1.0)
            result = sf.gaussian(result, sigma=sigma)
        elif filter_name == 'median':
            radius = params.get('radius', 3)
            selem = disk(radius)
            result = sf.median(result, selem=selem)
        elif filter_name == 'threshold_otsu':
            thresh = sf.threshold_otsu(result)
            result = result > thresh
        elif filter_name == 'threshold_local':
            block_size = params.get('block_size', 35)
            method = params.get('method', 'gaussian')
            offset = params.get('offset', 0)
            thresh = sf.threshold_local(result, block_size=block_size, method=method, offset=offset)
            result = result > thresh
            
        if verbose:
            plt.subplot(1, len(filters)+1, i+2)
            plt.imshow(result)
            plt.title(f'{filter_name}')
            plt.axis('off')
    
    if verbose:
        plt.tight_layout()
        plt.show()
        
    return result


def Assemble_Cell_Images(cell_images, layout=(5, 10), cell_size=None, normalize=True, pad=1, 
                         bg_color=0, border_color=0.2, verbose=False):
    """
    Assemble multiple cell images into a grid
    
    Parameters:
    -----------
    cell_images : list of numpy arrays
        List of cell images to assemble
    layout : tuple
        (rows, cols) layout of the grid
    cell_size : int or None
        Size to resize cells to. If None, use size of first cell.
    normalize : bool
        Whether to normalize each cell image
    pad : int
        Padding between cells
    bg_color : float
        Background color (0-1)
    border_color : float
        Border color (0-1)
    verbose : bool
        Whether to show progress
        
    Returns:
    --------
    numpy array
        Assembled grid of cell images
    """
    from skimage.transform import resize
    
    rows, cols = layout
    n_cells = min(len(cell_images), rows * cols)
    
    # Determine cell size
    if cell_size is None and len(cell_images) > 0:
        if len(cell_images[0].shape) == 3:  # Multi-channel
            cell_size = max(cell_images[0].shape[1], cell_images[0].shape[2])
        else:  # Single channel
            cell_size = max(cell_images[0].shape[0], cell_images[0].shape[1])
    
    # Create empty grid with padding
    total_height = rows * (cell_size + pad) + pad
    total_width = cols * (cell_size + pad) + pad
    
    # Determine number of channels
    if len(cell_images) > 0 and len(cell_images[0].shape) == 3:
        n_channels = cell_images[0].shape[0]
        grid = np.ones((n_channels, total_height, total_width)) * bg_color
    else:
        grid = np.ones((total_height, total_width)) * bg_color
    
    # Place each cell in the grid
    for i in range(n_cells):
        row = i // cols
        col = i % cols
        
        cell = cell_images[i]
        
        # Normalize if requested
        if normalize and np.max(cell) > np.min(cell):
            cell = (cell - np.min(cell)) / (np.max(cell) - np.min(cell))
        
        # Resize to square
        if len(cell.shape) == 3:  # Multi-channel
            n_ch, h, w = cell.shape
            
            if h != cell_size or w != cell_size:
                resized_cell = np.zeros((n_ch, cell_size, cell_size))
                for c in range(n_ch):
                    resized_cell[c] = resize(cell[c], (cell_size, cell_size), 
                                            preserve_range=True, anti_aliasing=True)
                cell = resized_cell
                
            # Add to grid
            y_start = row * (cell_size + pad) + pad
            x_start = col * (cell_size + pad) + pad
            
            # Add border by filling padding with border_color
            y_border = range(y_start - pad, y_start + cell_size + pad)
            x_border = range(x_start - pad, x_start + cell_size + pad)
            
            for c in range(n_channels):
                grid[c, y_border[0]:y_border[-1]+1, x_start-pad:x_start+cell_size+pad] = border_color
                grid[c, y_start-pad:y_start+cell_size+pad, x_border[0]:x_border[-1]+1] = border_color
                
                # Add cell
                grid[c, y_start:y_start+cell_size, x_start:x_start+cell_size] = cell[c]
                
        else:  # Single channel
            h, w = cell.shape
            
            if h != cell_size or w != cell_size:
                cell = resize(cell, (cell_size, cell_size), 
                              preserve_range=True, anti_aliasing=True)
                
            # Add to grid
            y_start = row * (cell_size + pad) + pad
            x_start = col * (cell_size + pad) + pad
            
            # Add border
            y_border = range(y_start - pad, y_start + cell_size + pad)
            x_border = range(x_start - pad, x_start + cell_size + pad)
            
            grid[y_border[0]:y_border[-1]+1, x_start-pad:x_start+cell_size+pad] = border_color
            grid[y_start-pad:y_start+cell_size+pad, x_border[0]:x_border[-1]+1] = border_color
            
            # Add cell
            grid[y_start:y_start+cell_size, x_start:x_start+cell_size] = cell
    
    return grid


def Save_Cell_Atlas(grid, output_path, title=None, dpi=150, format='png'):
    """
    Save a cell atlas grid to a file
    
    Parameters:
    -----------
    grid : numpy array
        Grid of assembled cells
    output_path : str
        Path to save the output file
    title : str or None
        Optional title to add to the image
    dpi : int
        Resolution for saving
    format : str
        File format ('png', 'jpg', 'pdf', etc.)
    """
    plt.figure(figsize=(grid.shape[1]/100, grid.shape[0]/100))
    
    if len(grid.shape) == 3:  # Multi-channel
        # Convert to RGB
        if grid.shape[0] == 1:  # Single channel to grayscale
            plt.imshow(grid[0], cmap='gray')
        elif grid.shape[0] == 2:  # Two channels as RG
            rgb = np.zeros((grid.shape[1], grid.shape[2], 3))
            rgb[:, :, 0] = grid[0]  # Red
            rgb[:, :, 1] = grid[1]  # Green
            plt.imshow(rgb)
        elif grid.shape[0] >= 3:  # Three+ channels as RGB
            rgb = np.zeros((grid.shape[1], grid.shape[2], 3))
            rgb[:, :, 0] = grid[0]  # Red
            rgb[:, :, 1] = grid[1]  # Green
            rgb[:, :, 2] = grid[2]  # Blue
            plt.imshow(rgb)
    else:  # Single channel
        plt.imshow(grid, cmap='gray')
    
    plt.axis('off')
    
    if title:
        plt.title(title, fontsize=12)
    
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=dpi, format=format, bbox_inches='tight')
    plt.close()
    
    print(f"Saved atlas to {output_path}")
    
    # Also save numpy array for further processing
    np_path = os.path.splitext(output_path)[0] + '.npy'
    np.save(np_path, grid)
    print(f"Saved numpy array to {np_path}")


if __name__ == "__main__":
    # Example usage
    print("This module provides functions for cell image analysis and atlas creation.")
    print("Import this module to use its functions in your scripts.")