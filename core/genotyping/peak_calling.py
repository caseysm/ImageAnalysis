"""Peak detection and analysis for fluorescence data."""

from typing import List, Optional, Tuple
import numpy as np
from scipy import ndimage, signal
from skimage import filters, measure

class PeakCaller:
    """Class for detecting and analyzing signal peaks in fluorescence data."""
    
    def __init__(
        self,
        min_peak_height: float = 0.2,
        min_peak_distance: int = 3,
        peak_width_range: Tuple[int, int] = (2, 10),
        threshold_std: float = 3.0
    ):
        """Initialize the peak caller.
        
        Args:
            min_peak_height: Minimum normalized peak height
            min_peak_distance: Minimum distance between peaks
            peak_width_range: (min_width, max_width) for valid peaks
            threshold_std: Standard deviations for thresholding
        """
        self.min_peak_height = min_peak_height
        self.min_peak_distance = min_peak_distance
        self.peak_width_range = peak_width_range
        self.threshold_std = threshold_std
        
    def find_peaks(
        self,
        signal_data: np.ndarray,
        background: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find peaks in fluorescence signal data.
        
        Args:
            signal_data: Raw fluorescence signal array
            background: Optional background signal to subtract
            
        Returns:
            Tuple of:
                - peak_positions: Array of peak positions
                - peak_heights: Array of peak heights
                - peak_widths: Array of peak widths
        """
        # Background subtraction if provided
        if background is not None:
            signal_data = signal_data - background
            
        # Normalize signal
        signal_norm = self._normalize_signal(signal_data)
        
        # Find peaks using scipy.signal
        peaks, properties = signal.find_peaks(
            signal_norm,
            height=self.min_peak_height,
            distance=self.min_peak_distance,
            width=self.peak_width_range
        )
        
        # Extract peak properties
        heights = properties['peak_heights']
        widths = properties['widths']
        
        return peaks, heights, widths
        
    def _normalize_signal(self, signal_data: np.ndarray) -> np.ndarray:
        """Normalize signal data.
        
        Args:
            signal_data: Raw signal array
            
        Returns:
            Normalized signal array
        """
        # Remove baseline
        baseline = ndimage.minimum_filter1d(signal_data, size=20)
        signal_baselined = signal_data - baseline
        
        # Scale to [0, 1]
        signal_norm = (signal_baselined - np.min(signal_baselined)) / \
                     (np.max(signal_baselined) - np.min(signal_baselined))
                     
        return signal_norm
        
    def calculate_quality_scores(
        self,
        peaks: np.ndarray,
        heights: np.ndarray,
        widths: np.ndarray
    ) -> np.ndarray:
        """Calculate quality scores for detected peaks.
        
        Args:
            peaks: Array of peak positions
            heights: Array of peak heights
            widths: Array of peak widths
            
        Returns:
            Array of quality scores
        """
        # Normalize metrics
        heights_norm = heights / np.max(heights)
        widths_norm = 1 - (widths - self.peak_width_range[0]) / \
                     (self.peak_width_range[1] - self.peak_width_range[0])
                     
        # Calculate scores as weighted combination
        scores = 0.7 * heights_norm + 0.3 * widths_norm
        
        return scores
        
    def call_bases(
        self,
        peaks: np.ndarray,
        heights: np.ndarray,
        widths: np.ndarray,
        channel_data: List[np.ndarray]
    ) -> Tuple[List[str], np.ndarray]:
        """Call bases from peak data across channels.
        
        Args:
            peaks: Array of peak positions
            heights: Array of peak heights
            widths: Array of peak widths
            channel_data: List of signal arrays for each channel
            
        Returns:
            Tuple of:
                - base_calls: List of called bases
                - quality_scores: Array of quality scores
        """
        base_map = ['A', 'C', 'G', 'T']
        n_channels = len(channel_data)
        
        if n_channels != len(base_map):
            raise ValueError(f"Expected {len(base_map)} channels, got {n_channels}")
            
        base_calls = []
        quality_scores = []
        
        # Process each peak position
        for peak_idx, peak_pos in enumerate(peaks):
            # Get intensities at peak position for each channel
            intensities = [channel[peak_pos] for channel in channel_data]
            
            # Find channel with maximum intensity
            max_channel = np.argmax(intensities)
            
            # Calculate quality score
            score = self.calculate_quality_scores(
                np.array([peak_pos]),
                np.array([heights[peak_idx]]),
                np.array([widths[peak_idx]])
            )[0]
            
            base_calls.append(base_map[max_channel])
            quality_scores.append(score)
            
        return base_calls, np.array(quality_scores)
        
    def filter_peaks(
        self,
        peaks: np.ndarray,
        heights: np.ndarray,
        widths: np.ndarray,
        quality_threshold: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter peaks based on quality scores.
        
        Args:
            peaks: Array of peak positions
            heights: Array of peak heights
            widths: Array of peak widths
            quality_threshold: Minimum quality score
            
        Returns:
            Tuple of filtered (peaks, heights, widths)
        """
        quality_scores = self.calculate_quality_scores(peaks, heights, widths)
        mask = quality_scores >= quality_threshold
        
        return peaks[mask], heights[mask], widths[mask] 