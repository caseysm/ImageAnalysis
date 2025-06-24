import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import skimage.io
import skimage.morphology
from scipy import ndimage as ndi
import decorator
import sys
import warnings

warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='regionprops and image moments')
warnings.filterwarnings('ignore', message='non-tuple sequence for multi')
warnings.filterwarnings('ignore', message='precision loss when converting')


@decorator.decorator
def applyIJ(f, arr, *args, **kwargs):
    """Apply a function that expects 2D input to the trailing two
    dimensions of an array. The function must output an array whose shape
    depends only on the input shape.
    """
    h, w = arr.shape[-2:]
    reshaped = arr.reshape((-1, h, w))

    # kwargs are not actually getting passed in?
    arr_ = [f(frame, *args, **kwargs) for frame in reshaped]

    output_shape = arr.shape[:-2] + arr_[0].shape
    return np.array(arr_).reshape(output_shape)


class Snake:
    """Container class for methods that act directly on data (names start with
    underscore) and methods that act on arguments from snakemake (e.g., filenames
    provided instead of image and table data). The snakemake methods (no underscore)
    are automatically loaded by `Snake.load_methods`.
    """

    # ALIGNMENT AND SEGMENTATION

    @staticmethod
    def remove_channels(data, remove_index):
        """Remove channel or list of channels from array of shape (..., CHANNELS, I, J).
        """
        channels_mask = np.ones(data.shape[-3], dtype=bool)
        channels_mask[remove_index] = False
        data = data[..., channels_mask, :, :]
        return data

    @staticmethod
    def _align_SBS(data, method='DAPI', upsample_factor=2, window=2, cutoff=1, align_channels=slice(1, None), keep_trailing=False):
        """Rigid alignment of sequencing cycles and channels.

        Parameters
        ----------

        data : numpy array
            Image data, expected dimensions of (CYCLE, CHANNEL, I, J).

        method : {'DAPI','SBS_mean'}, default 'DAPI'
            Method for aligning 'data' across cycles. 'DAPI' uses cross-correlation between subsequent cycles
            of DAPI images, assumes sequencing channels are aligned to DAPI images. 'SBS_mean' uses the
            mean background signal from the SBS channels to determine image offsets between cycles of imaging,
            again using cross-correlation.

        upsample_factor : int, default 2
            Subpixel alignment is done if `upsample_factor` is greater than one (can be slow).
            Parameter passed to skimage.feature.register_translation.

        window : int, default 2
            A centered subset of data is used if `window` is greater than one. The size of the removed border is
            int((x/2.) * (1 - 1/float(window))).

        cutoff : float, default 1
            Threshold for removing extreme values from SBS channels when using method='SBS_mean'. Channels are normalized
            to the 70th percentile, and normalized values greater than `cutoff` are replaced by `cutoff`.

        align_channels : slice object or None, default slice(1,None)
            If not None, aligns channels (defined by the passed slice object) to each other within each cycle. If
            None, does not align channels within cycles. Useful in particular for cases where images for all stage
            positions are acquired for one SBS channel at a time, i.e., acquisition order of channels(positions).

        keep_trailing : boolean, default True
            If True, keeps only the minimum number of trailing channels across cycles. E.g., if one cycle contains 6 channels,
            but all others have 5, only uses trailing 5 channels for alignment.

        n : int, default 1
            The first SBS channel in `data`.

        Returns
        -------

        aligned : numpy array
            Aligned image data, same dimensions as `data` unless `data` contained different numbers of channels between cycles
            and keep_trailing=True.
        """
        data = np.array(data)
        if keep_trailing:
            min_channels = min(d.shape[0] for d in data)
            data = np.array([cycle[-min_channels:] for cycle in data])

        h, w = data.shape[-2:]
        border_h = int((h / 2) * (1 - 1 / float(window)))
        border_w = int((w / 2) * (1 - 1 / float(window)))
        border = (slice(border_h, -border_h), slice(border_w, -border_w))

        # use phase cross-correlation for subpixel registration
        from skimage.registration import phase_cross_correlation
        def register_translation(moving, fixed, upsample_factor=1):
            return phase_cross_correlation(fixed, moving, upsample_factor=upsample_factor)[0]

        if method == 'DAPI':
            dapi_reference = []
            dapi_moving = []
            
            # DAPI channel is at index 0
            for i in range(data.shape[0]):
                dapi_reference.append(data[i, 0][border])
            
            # use each cycle as reference for the next
            shifts_between_cycles = []
            for i in range(1, len(dapi_reference)):
                shift = register_translation(dapi_reference[i], dapi_reference[i-1], upsample_factor)
                shifts_between_cycles.append(shift)

            # calculate cumulative shift
            s_y, s_x = np.cumsum(np.array(shifts_between_cycles), axis=0).T
            s_y, s_x = np.hstack([[0, 0], [s_y, s_x]]).T
            
            from scipy.ndimage import shift as shift_image
            
            # apply shifts
            aligned = []
            for i, cycle in enumerate(data):
                shifted_cycle = []
                for channel in cycle:
                    shifted_channel = shift_image(channel, (s_y[i], s_x[i]), order=1, mode='constant', cval=0)
                    shifted_cycle.append(shifted_channel)
                aligned.append(shifted_cycle)
            
            aligned = np.array(aligned)
            
            # Align channels within each cycle if requested
            if align_channels is not None:
                for i, cycle in enumerate(aligned):
                    reference = cycle[0]  # Use DAPI as reference
                    for j in range(1, cycle.shape[0]):
                        shift = register_translation(cycle[j], reference, upsample_factor)
                        aligned[i, j] = shift_image(cycle[j], shift, order=1, mode='constant', cval=0)
            
            return aligned
            
        elif method == 'SBS_mean':
            # Calculate mean of SBS channels for each cycle, ignoring extreme values
            sbs_means = []
            for i, cycle in enumerate(data):
                sbs_channels = cycle[1:] if cycle.shape[0] > 1 else cycle
                normalized = []
                for channel in sbs_channels:
                    # Normalize to 70th percentile and clip to cutoff
                    norm = channel / np.percentile(channel, 70)
                    norm[norm > cutoff] = cutoff
                    normalized.append(norm)
                sbs_mean = np.mean(normalized, axis=0)
                sbs_means.append(sbs_mean[border])
            
            # Calculate shifts between consecutive cycles
            shifts_between_cycles = []
            for i in range(1, len(sbs_means)):
                shift = register_translation(sbs_means[i], sbs_means[i-1], upsample_factor)
                shifts_between_cycles.append(shift)
                
            # Calculate cumulative shifts
            s_y, s_x = np.cumsum(np.array(shifts_between_cycles), axis=0).T
            s_y, s_x = np.hstack([[0, 0], [s_y, s_x]]).T
            
            # Apply shifts
            from scipy.ndimage import shift as shift_image
            aligned = []
            for i, cycle in enumerate(data):
                shifted_cycle = []
                for channel in cycle:
                    shifted_channel = shift_image(channel, (s_y[i], s_x[i]), order=1, mode='constant', cval=0)
                    shifted_cycle.append(shifted_channel)
                aligned.append(shifted_cycle)
                
            aligned = np.array(aligned)
            
            # Align channels within each cycle if requested
            if align_channels is not None:
                for i, cycle in enumerate(aligned):
                    reference = cycle[0]  # Use DAPI as reference
                    for j in range(1, cycle.shape[0]):
                        shift = register_translation(cycle[j], reference, upsample_factor)
                        aligned[i, j] = shift_image(cycle[j], shift, order=1, mode='constant', cval=0)
            
            return aligned
        else:
            raise ValueError(f"Unknown alignment method: {method}")

    @staticmethod
    def _transform_log(data, skip_index=None):
        """Apply Laplacian of Gaussian transform to channels.

        Parameters
        ----------
        data : numpy array
            Image data, dimensions of (CYCLE, CHANNEL, I, J).
        skip_index : int or list of ints or None, default None
            Skip transform for these channel indices.

        Returns
        -------
        transformed : numpy array
            Transformed image data, same dimensions as `data`.
        """
        transformed_data = data.copy()
        
        if skip_index is None:
            skip_index = []
        elif isinstance(skip_index, int):
            skip_index = [skip_index]
        
        cycles, channels, height, width = data.shape
        
        for cy in range(cycles):
            for ch in range(channels):
                if ch in skip_index:
                    continue
                    
                # Apply Laplacian of Gaussian transform
                from scipy import ndimage
                transformed_data[cy, ch] = ndimage.gaussian_laplace(data[cy, ch], sigma=1)
                
                # Ensure positive values (optional)
                transformed_data[cy, ch] = np.abs(transformed_data[cy, ch])
        
        return transformed_data

    @staticmethod
    def _max_filter(data, size=3, remove_index=None):
        """Apply maximum filter to channels.

        Parameters
        ----------
        data : numpy array
            Image data, dimensions of (CYCLE, CHANNEL, I, J).
        size : int, default 3
            Size of the maximum filter.
        remove_index : int or list of ints or None, default None
            Remove these channel indices.

        Returns
        -------
        filtered : numpy array
            Filtered image data, dimensions of (CYCLE, CHANNEL, I, J).
        """
        from scipy import ndimage
        
        # Handle the remove_index parameter
        if remove_index is not None:
            if isinstance(remove_index, int):
                remove_index = [remove_index]
            channels_mask = np.ones(data.shape[1], dtype=bool)
            channels_mask[remove_index] = False
            filtered_data = data[:, channels_mask, :, :]
        else:
            filtered_data = data.copy()
        
        cycles, channels, height, width = filtered_data.shape
        
        # Apply maximum filter to each channel
        for cy in range(cycles):
            for ch in range(channels):
                filtered_data[cy, ch] = ndimage.maximum_filter(filtered_data[cy, ch], size=size)
        
        return filtered_data

    @staticmethod
    def _compute_std(data, remove_index=None):
        """Compute standard deviation across cycles for each pixel.

        Parameters
        ----------
        data : numpy array
            Image data, dimensions of (CYCLE, CHANNEL, I, J).
        remove_index : int or list of ints or None, default None
            Remove these channel indices.

        Returns
        -------
        std : numpy array
            Standard deviation of data across cycles, dimensions of (I, J).
        """
        # Handle the remove_index parameter
        if remove_index is not None:
            if isinstance(remove_index, int):
                remove_index = [remove_index]
            channels_mask = np.ones(data.shape[1], dtype=bool)
            channels_mask[remove_index] = False
            filtered_data = data[:, channels_mask, :, :]
        else:
            filtered_data = data
        
        # Reshape data to combine cycle and channel dimensions
        cycles, channels, height, width = filtered_data.shape
        reshaped_data = filtered_data.reshape(cycles * channels, height, width)
        
        # Compute standard deviation across the first dimension (combined cycles and channels)
        std = np.std(reshaped_data, axis=0)
        
        return std

    @staticmethod
    def _find_peaks(std_data, threshold_abs=None, threshold_rel=0.2, min_distance=5):
        """Find peaks in standard deviation data.

        Parameters
        ----------
        std_data : numpy array
            Standard deviation data, dimensions of (I, J).
        threshold_abs : float or None, default None
            Minimum intensity of peaks. If None, uses threshold_rel.
        threshold_rel : float, default 0.2
            Minimum intensity of peaks, relative to the maximum intensity of std_data.
        min_distance : int, default 5
            Minimum distance between peaks.

        Returns
        -------
        peaks : numpy array
            Binary mask indicating peak locations, dimensions of (I, J).
        """
        from skimage.feature import peak_local_max
        
        # Find local maxima
        if threshold_abs is None:
            threshold_abs = threshold_rel * np.max(std_data)
            
        coordinates = peak_local_max(std_data, 
                                    min_distance=min_distance, 
                                    threshold_abs=threshold_abs)
        
        # Create binary mask
        peaks = np.zeros_like(std_data, dtype=bool)
        for coord in coordinates:
            peaks[coord[0], coord[1]] = True
        
        return peaks


class Utils:
    """Common utility functions."""

    @staticmethod
    def match_size(source, target):
        """Resize source to match target dimensions while preserving data range and type.

        Parameters
        ----------
        source : numpy array
            Source data to resize
        target : numpy array
            Target data whose dimensions to match

        Returns
        -------
        resized : numpy array
            Resized source data with dimensions matching target
        """
        from skimage.transform import resize
        
        # Get target shape
        target_shape = target.shape
        
        # Remember original min and max values
        source_min = np.min(source)
        source_max = np.max(source)
        
        # Resize to target shape
        resized = resize(source, target_shape, mode='edge', preserve_range=True)
        
        # Restore original data range
        if source_min != source_max:
            resized = (resized - np.min(resized)) / (np.max(resized) - np.min(resized))
            resized = source_min + resized * (source_max - source_min)
        
        # Convert back to original dtype
        resized = resized.astype(source.dtype)
        
        return resized


class Align:
    """Image alignment utilities."""
    
    @staticmethod
    def normalize_image(image, percentile=99):
        """Normalize image to [0,1] range based on percentile."""
        low = np.min(image)
        high = np.percentile(image, percentile)
        if high > low:
            norm = (image - low) / (high - low)
            norm[norm > 1] = 1
            norm[norm < 0] = 0
            return norm
        return np.zeros_like(image)
    
    @staticmethod
    def filter_image(image, sigma=1):
        """Apply Gaussian filter to image."""
        from scipy import ndimage
        return ndimage.gaussian_filter(image, sigma=sigma)
    
    @staticmethod
    def calculate_offset(fixed, moving, upsample_factor=4):
        """Calculate offset between fixed and moving images."""
        from skimage.registration import phase_cross_correlation
        shift, error, diffphase = phase_cross_correlation(fixed, moving, upsample_factor=upsample_factor)
        return shift
    
    @staticmethod
    def apply_offset(image, offset):
        """Apply offset to image."""
        from scipy.ndimage import shift as shift_image
        return shift_image(image, offset, order=1, mode='constant', cval=0)
    
    @staticmethod
    def align_within_cycle(cycle_data, reference_channel=0, upsample_factor=4):
        """Align channels within a cycle to reference channel."""
        aligned = np.zeros_like(cycle_data)
        aligned[reference_channel] = cycle_data[reference_channel]
        
        for i in range(cycle_data.shape[0]):
            if i != reference_channel:
                fixed = Align.normalize_image(cycle_data[reference_channel])
                moving = Align.normalize_image(cycle_data[i])
                
                offset = Align.calculate_offset(fixed, moving, upsample_factor)
                aligned[i] = Align.apply_offset(cycle_data[i], offset)
                
        return aligned


class QC:
    """Quality control and visualization utilities."""
    
    @staticmethod
    def plot_mapping_rates(report_df, title='Mapping Rates'):
        """Plot mapping rates from report dataframe."""
        plt.figure(figsize=(10, 6))
        
        if 'assigned' in report_df.columns:
            assigned = report_df['assigned'].sum()
            total = len(report_df)
            rate = assigned / total * 100 if total > 0 else 0
            
            plt.bar(['Assigned', 'Unassigned'], [rate, 100-rate])
            plt.ylabel('Percentage (%)')
            plt.title(f"{title}: {assigned}/{total} ({rate:.1f}%)")
            
        return plt.gcf()
    
    @staticmethod
    def plot_barcode_heatmap(counts_df, title='Barcode Counts'):
        """Create heatmap of barcode counts."""
        plt.figure(figsize=(12, 10))
        
        if len(counts_df) > 0:
            pivot_data = counts_df.pivot_table(index='gene', columns='barcode', values='count', fill_value=0)
            
            with sns.axes_style("white"):
                sns.heatmap(np.log1p(pivot_data), cmap='viridis', 
                           xticklabels=True, yticklabels=True)
                
            plt.title(title)
            plt.tight_layout()
            
        return plt.gcf()