"""Barcode assignment and matching functionality."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from Bio import pairwise2
from Bio.Seq import Seq

class BarcodeAssigner:
    """Class for assigning called bases to known barcodes."""
    
    def __init__(
        self,
        barcode_library: pd.DataFrame,
        max_hamming_distance: int = 1,
        min_quality_score: float = 0.8
    ):
        """Initialize the barcode assigner.
        
        Args:
            barcode_library: DataFrame containing barcode sequences and metadata
            max_hamming_distance: Maximum allowed Hamming distance for matches
            min_quality_score: Minimum quality score for base calls
        """
        self.barcode_library = barcode_library
        self.max_hamming_distance = max_hamming_distance
        self.min_quality_score = min_quality_score
        
        # Create lookup dictionaries
        self.sequence_to_id = dict(zip(
            barcode_library['sequence'],
            barcode_library['barcode_id']
        ))
        self.id_to_gene = dict(zip(
            barcode_library['barcode_id'],
            barcode_library['gene']
        ))
        
    def assign_barcodes(
        self,
        base_calls: List[str],
        quality_scores: np.ndarray
    ) -> Tuple[str, float, bool]:
        """Assign base calls to a barcode.
        
        Args:
            base_calls: List of called bases
            quality_scores: Array of quality scores
            
        Returns:
            Tuple of:
                - assigned_barcode: Barcode ID or 'unassigned'
                - match_score: Score for the assignment
                - is_ambiguous: Whether the assignment is ambiguous
        """
        # Filter low quality bases
        high_quality_mask = quality_scores >= self.min_quality_score
        if not np.any(high_quality_mask):
            return 'unassigned', 0.0, False
            
        # Combine bases into sequence
        sequence = ''.join(base_calls)
        
        # Find best matches
        matches = self._find_matches(sequence)
        
        if not matches:
            return 'unassigned', 0.0, False
            
        # Check for ambiguity
        if len(matches) > 1:
            return 'ambiguous', matches[0][1], True
            
        return matches[0][0], matches[0][1], False
        
    def _find_matches(
        self,
        sequence: str
    ) -> List[Tuple[str, float]]:
        """Find matching barcodes for a sequence.
        
        Args:
            sequence: Query sequence
            
        Returns:
            List of (barcode_id, match_score) tuples
        """
        matches = []
        
        for ref_seq, barcode_id in self.sequence_to_id.items():
            # Calculate Hamming distance
            distance = self._hamming_distance(sequence, ref_seq)
            
            if distance <= self.max_hamming_distance:
                # Convert distance to similarity score
                score = 1 - (distance / len(sequence))
                matches.append((barcode_id, score))
                
        # Sort by score
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
        
    @staticmethod
    def _hamming_distance(s1: str, s2: str) -> int:
        """Calculate Hamming distance between two sequences.
        
        Args:
            s1: First sequence
            s2: Second sequence
            
        Returns:
            Hamming distance
        """
        if len(s1) != len(s2):
            return float('inf')
            
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))
        
    def get_gene_for_barcode(self, barcode_id: str) -> Optional[str]:
        """Get gene name for a barcode ID.
        
        Args:
            barcode_id: Barcode identifier
            
        Returns:
            Gene name or None if not found
        """
        return self.id_to_gene.get(barcode_id)
        
    def assign_cell_barcodes(
        self,
        cell_data: Dict[int, Dict[str, any]]
    ) -> pd.DataFrame:
        """Assign barcodes to cells.
        
        Args:
            cell_data: Dictionary mapping cell IDs to their data
                Each cell should have 'base_calls' and 'quality_scores'
                
        Returns:
            DataFrame with cell barcode assignments
        """
        assignments = []
        
        for cell_id, data in cell_data.items():
            base_calls = data.get('base_calls', [])
            quality_scores = data.get('quality_scores', np.array([]))
            
            if not base_calls or len(quality_scores) == 0:
                continue
                
            barcode_id, score, is_ambiguous = self.assign_barcodes(
                base_calls,
                quality_scores
            )
            
            gene_name = self.get_gene_for_barcode(barcode_id) if barcode_id not in ['unassigned', 'ambiguous'] else None
            
            assignments.append({
                'cell_id': cell_id,
                'barcode_id': barcode_id,
                'gene': gene_name,
                'match_score': score,
                'is_ambiguous': is_ambiguous,
                'mean_quality': np.mean(quality_scores)
            })
            
        return pd.DataFrame(assignments)
        
    def summarize_assignments(self, assignments: pd.DataFrame) -> Dict[str, int]:
        """Generate summary statistics for barcode assignments.
        
        Args:
            assignments: DataFrame of cell barcode assignments
            
        Returns:
            Dictionary of summary statistics
        """
        total_cells = len(assignments)
        assigned = assignments['barcode_id'].notna().sum()
        unambiguous = (~assignments['is_ambiguous']).sum()
        ambiguous = assignments['is_ambiguous'].sum()
        unassigned = (assignments['barcode_id'] == 'unassigned').sum()
        
        unique_genes = assignments['gene'].nunique()
        
        return {
            'total_cells': total_cells,
            'assigned_cells': assigned,
            'unambiguous_assignments': unambiguous,
            'ambiguous_assignments': ambiguous,
            'unassigned_cells': unassigned,
            'unique_genes': unique_genes
        } 