#!/usr/bin/env python3
"""
Segmentation Comparison Test.

This script compares the segmentation output from the modified original pipeline
and the refactored imageanalysis package.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import json
import shutil
import subprocess
import time
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up logging
def setup_logging(log_dir):
    """Set up logging for the test script."""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"segmentation_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("segmentation_comparison")

class SegmentationComparison:
    """Class to compare the segmentation between modified original and refactored pipelines."""
    
    def __init__(self, args):
        """Initialize the comparison with command-line arguments."""
        self.data_dir = Path(args.data_dir)
        self.modified_output_dir = Path(args.modified_output_dir)
        self.refactored_output_dir = Path(args.refactored_output_dir)
        self.wells = args.wells
        self.timeout = args.timeout
        self.modified_script_dir = Path(args.modified_script_dir)
        
        # Create output directories
        os.makedirs(self.modified_output_dir, exist_ok=True)
        os.makedirs(self.refactored_output_dir, exist_ok=True)
        
        # Set up logging
        self.logger = setup_logging(args.log_dir)
        
        # Find input files
        self.phenotyping_files = list(self.data_dir.glob('phenotyping/*.nd2'))
        self.logger.info(f"Found {len(self.phenotyping_files)} phenotyping files")
        
        # Limit input files to speed up testing
        self.input_limit = args.input_limit
        if self.input_limit > 0 and self.input_limit < len(self.phenotyping_files):
            self.phenotyping_files = self.phenotyping_files[:self.input_limit]
            self.logger.info(f"Limited to first {self.input_limit} phenotyping files")
        
        # Check for the modified original pipeline scripts
        if not os.path.exists(self.modified_script_dir):
            self.logger.error(f"Modified script directory not found: {self.modified_script_dir}")
            raise FileNotFoundError(f"Modified script directory not found: {self.modified_script_dir}")
        
        # Check for required scripts
        required_scripts = ["Segment_10X.py"]
        for script in required_scripts:
            if not os.path.exists(os.path.join(self.modified_script_dir, script)):
                self.logger.error(f"Required script not found: {script}")
                raise FileNotFoundError(f"Required script not found: {script}")

    def _run_command(self, cmd, description=None):
        """Run a command with timeout and logging."""
        if description:
            self.logger.info(f"Running: {description}")
        self.logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                self.logger.info(f"Command completed successfully in {elapsed:.2f} seconds")
            else:
                self.logger.error(f"Command failed with return code {result.returncode}")
                self.logger.error(f"STDERR: {result.stderr}")
                
            return result.returncode == 0
        
        except subprocess.TimeoutExpired:
            self.logger.error(f"Command timed out after {self.timeout} seconds")
            return False
        
        except Exception as e:
            self.logger.error(f"Error running command: {e}")
            return False
    
    def run_modified_segmentation(self):
        """Run the modified original segmentation pipeline."""
        self.logger.info("=== Running Modified Original Segmentation ===")
        
        # Create output directory
        segmentation_dir = os.path.join(self.modified_output_dir, 'segmentation')
        os.makedirs(segmentation_dir, exist_ok=True)
        
        # Track success status
        success = True
        
        # Use the modified original Segment_10X.py script
        for pheno_file in self.phenotyping_files:
            # Extract well name from filename
            filename = pheno_file.name
            well_match = filename.split('_')[0]
            well = well_match if well_match.startswith('Well') else 'Well1'
            
            if self.wells and well not in self.wells:
                continue
            
            # Create well directory
            well_dir = os.path.join(segmentation_dir, well)
            os.makedirs(well_dir, exist_ok=True)
            
            # Run the script
            # Note: Segment_10X.py expects a positional input_file argument
            cmd = [
                sys.executable,
                os.path.join(self.modified_script_dir, "Segment_10X.py"),
                str(pheno_file),
                "--output-dir", well_dir,
                "--nuc-channel", "0",  # Assuming DAPI is first channel
                "--cell-channel", "1"  # Assuming mClov3 is second channel
            ]
            
            cmd_success = self._run_command(cmd, f"Segmentation for {well} - {filename}")
            if not cmd_success:
                self.logger.error(f"Segmentation failed for {well} - {filename}")
                success = False
        
        return success
    
    def run_refactored_segmentation(self):
        """Run the refactored segmentation pipeline."""
        self.logger.info("=== Running Refactored Segmentation ===")
        
        # Create output directory
        segmentation_dir = os.path.join(self.refactored_output_dir, 'segmentation')
        os.makedirs(segmentation_dir, exist_ok=True)
        
        # Track success status
        success = True
        
        # Use the refactored package script
        for pheno_file in self.phenotyping_files:
            # Extract well name from filename
            filename = pheno_file.name
            well_match = filename.split('_')[0]
            well = well_match if well_match.startswith('Well') else 'Well1'
            
            if self.wells and well not in self.wells:
                continue
            
            # Run the script with the correct parameters
            cmd = [
                sys.executable,
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "bin", "run_segmentation.py"),
                str(pheno_file),
                "--magnification", "10x",  # Required parameter
                "--output-dir", segmentation_dir,
                "--wells", well,
                "--nuclear-channel", "0",
                "--cell-channel", "1"
            ]
            
            cmd_success = self._run_command(cmd, f"Segmentation for {well} - {filename}")
            if not cmd_success:
                self.logger.error(f"Segmentation failed for {well} - {filename}")
                success = False
        
        return success
    
    def compare_segmentation_outputs(self):
        """Compare segmentation results between modified original and refactored pipelines."""
        self.logger.info("=== Comparing Segmentation Outputs ===")
        
        results = {
            'match': True,
            'differences': []
        }
        
        modified_dir = os.path.join(self.modified_output_dir, 'segmentation')
        refactored_dir = os.path.join(self.refactored_output_dir, 'segmentation')
        
        # Check if directories exist
        if not os.path.exists(modified_dir) or not os.path.exists(refactored_dir):
            results['match'] = False
            results['differences'].append("Segmentation directory missing in one of the outputs")
            return results
        
        # Compare well directories
        modified_wells = set(d.name for d in Path(modified_dir).iterdir() if d.is_dir())
        refactored_wells = set(d.name for d in Path(refactored_dir).iterdir() if d.is_dir())
        
        if modified_wells != refactored_wells:
            results['match'] = False
            results['differences'].append(
                f"Different wells found: modified {modified_wells}, refactored {refactored_wells}"
            )
        
        # Compare masks for common wells
        common_wells = modified_wells.intersection(refactored_wells)
        for well in common_wells:
            # Check nuclear masks
            modified_nuc_path = self._find_nuclear_mask(os.path.join(modified_dir, well))
            refactored_nuc_path = self._find_nuclear_mask(os.path.join(refactored_dir, well))
            
            if modified_nuc_path and refactored_nuc_path:
                try:
                    modified_nuc = np.load(modified_nuc_path)
                    refactored_nuc = np.load(refactored_nuc_path)
                    
                    # Handle the case where modified_nuc is a tuple (from original format)
                    if isinstance(modified_nuc, tuple) and len(modified_nuc) == 2:
                        # Original format stores (image, mask)
                        modified_nuc = modified_nuc[1]
                    
                    # Compare shapes
                    if modified_nuc.shape != refactored_nuc.shape:
                        results['match'] = False
                        results['differences'].append(
                            f"{well} nuclear mask shape mismatch: {modified_nuc.shape} vs {refactored_nuc.shape}"
                        )
                        continue
                    
                    # Compare number of nuclei
                    modified_count = len(np.unique(modified_nuc)) - 1  # Subtract background
                    refactored_count = len(np.unique(refactored_nuc)) - 1
                    
                    diff_percentage = 0
                    if max(modified_count, refactored_count) > 0:
                        diff_percentage = abs(modified_count - refactored_count) / max(modified_count, refactored_count) * 100
                    
                    if diff_percentage > 20:  # Allow up to 20% difference
                        results['match'] = False
                        results['differences'].append(
                            f"{well} nuclear count mismatch: {modified_count} vs {refactored_count} ({diff_percentage:.1f}% difference)"
                        )
                    else:
                        self.logger.info(f"{well} nuclear counts: {modified_count} vs {refactored_count} ({diff_percentage:.1f}% difference)")
                        
                except Exception as e:
                    results['match'] = False
                    results['differences'].append(f"Error comparing {well} nuclear masks: {str(e)}")
            else:
                results['match'] = False
                results['differences'].append(f"{well} nuclear mask missing in one of the outputs")
            
            # Check cell masks
            modified_cell_path = self._find_cell_mask(os.path.join(modified_dir, well))
            refactored_cell_path = self._find_cell_mask(os.path.join(refactored_dir, well))
            
            if modified_cell_path and refactored_cell_path:
                try:
                    modified_cell = np.load(modified_cell_path)
                    refactored_cell = np.load(refactored_cell_path)
                    
                    # Handle the case where modified_cell is a tuple (from original format)
                    if isinstance(modified_cell, tuple) and len(modified_cell) == 2:
                        # Original format stores (image, mask)
                        modified_cell = modified_cell[1]
                    
                    # Compare shapes
                    if modified_cell.shape != refactored_cell.shape:
                        results['match'] = False
                        results['differences'].append(
                            f"{well} cell mask shape mismatch: {modified_cell.shape} vs {refactored_cell.shape}"
                        )
                        continue
                    
                    # Compare number of cells
                    modified_count = len(np.unique(modified_cell)) - 1  # Subtract background
                    refactored_count = len(np.unique(refactored_cell)) - 1
                    
                    diff_percentage = 0
                    if max(modified_count, refactored_count) > 0:
                        diff_percentage = abs(modified_count - refactored_count) / max(modified_count, refactored_count) * 100
                    
                    if diff_percentage > 20:  # Allow up to 20% difference
                        results['match'] = False
                        results['differences'].append(
                            f"{well} cell count mismatch: {modified_count} vs {refactored_count} ({diff_percentage:.1f}% difference)"
                        )
                    else:
                        self.logger.info(f"{well} cell counts: {modified_count} vs {refactored_count} ({diff_percentage:.1f}% difference)")
                        
                except Exception as e:
                    results['match'] = False
                    results['differences'].append(f"Error comparing {well} cell masks: {str(e)}")
            else:
                results['match'] = False
                results['differences'].append(f"{well} cell mask missing in one of the outputs")
        
        return results
    
    def _find_nuclear_mask(self, directory):
        """Find nuclear mask file in the given directory."""
        # Try standard name
        standard_path = os.path.join(directory, "nuclei_mask.npy")
        if os.path.exists(standard_path):
            return standard_path
        
        # Try specific file pattern matching
        files = list(Path(directory).glob("*nuclei_mask.npy"))
        if files:
            return str(files[0])
            
        # Check for original format
        nuc_dirs = list(Path(directory).glob("nucs/well_*/Seg_Nuc-*.npy"))
        if nuc_dirs:
            return str(nuc_dirs[0])
            
        # Check the directory itself for any npy files
        npy_files = list(Path(directory).glob("*.npy"))
        for npy_file in npy_files:
            if "nuc" in npy_file.name.lower():
                return str(npy_file)
                
        return None
    
    def _find_cell_mask(self, directory):
        """Find cell mask file in the given directory."""
        # Try standard name
        standard_path = os.path.join(directory, "cell_mask.npy")
        if os.path.exists(standard_path):
            return standard_path
        
        # Try specific file pattern matching
        files = list(Path(directory).glob("*cell_mask.npy"))
        if files:
            return str(files[0])
            
        # Check for original format
        cell_dirs = list(Path(directory).glob("cells/well_*/Seg_Cells-*.npy"))
        if cell_dirs:
            return str(cell_dirs[0])
            
        # Check the directory itself for any npy files
        npy_files = list(Path(directory).glob("*.npy"))
        for npy_file in npy_files:
            if "cell" in npy_file.name.lower():
                return str(npy_file)
                
        return None
    
    def run_comparison(self):
        """Run the full segmentation comparison."""
        self.logger.info("Starting segmentation comparison test")
        self.logger.info(f"Data directory: {self.data_dir}")
        self.logger.info(f"Modified original output: {self.modified_output_dir}")
        self.logger.info(f"Refactored output: {self.refactored_output_dir}")
        self.logger.info(f"Modified script directory: {self.modified_script_dir}")
        
        # Run modified original segmentation
        modified_success = self.run_modified_segmentation()
        if not modified_success:
            self.logger.error("Modified original segmentation failed")
            return False
        
        # Run refactored segmentation
        refactored_success = self.run_refactored_segmentation()
        if not refactored_success:
            self.logger.error("Refactored segmentation failed")
            return False
        
        # Compare outputs
        compare_results = self.compare_segmentation_outputs()
        
        # Create summary report
        report_dir = os.path.join(os.path.dirname(self.modified_output_dir), "report")
        os.makedirs(report_dir, exist_ok=True)
        
        report_file = os.path.join(report_dir, f"segmentation_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_directory': str(self.data_dir),
            'modified_output_directory': str(self.modified_output_dir),
            'refactored_output_directory': str(self.refactored_output_dir),
            'wells_tested': self.wells,
            'overall_match': compare_results['match'],
            'differences': compare_results['differences']
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        self.logger.info(f"Report saved to {report_file}")
        
        return compare_results['match']

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare segmentation between modified original pipeline and refactored package"
    )
    
    parser.add_argument(
        '--data-dir',
        default='/home/casey/Desktop/ShalemLab/ImageAnalysis/data',
        help="Data directory containing test images"
    )
    
    parser.add_argument(
        '--modified-output-dir',
        default='/home/casey/Desktop/ShalemLab/ImageAnalysis/results/segmentation_comparison/modified',
        help="Output directory for modified original pipeline"
    )
    
    parser.add_argument(
        '--refactored-output-dir',
        default='/home/casey/Desktop/ShalemLab/ImageAnalysis/results/segmentation_comparison/refactored',
        help="Output directory for refactored pipeline"
    )
    
    parser.add_argument(
        '--log-dir',
        default='/home/casey/Desktop/ShalemLab/ImageAnalysis/results/segmentation_comparison/logs',
        help="Directory for log files"
    )
    
    parser.add_argument(
        '--modified-script-dir',
        default='/home/casey/Desktop/ShalemLab/ImageAnalysis/modified_original_pipeline',
        help="Directory containing the modified original pipeline scripts"
    )
    
    parser.add_argument(
        '--wells',
        nargs='+',
        default=['Well1'],
        help="Wells to process (default: Well1)"
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=600,
        help="Timeout for commands in seconds (default: 600)"
    )
    
    parser.add_argument(
        '--input-limit',
        type=int,
        default=2,
        help="Limit the number of input files to process (default: 2)"
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directories
    os.makedirs(os.path.dirname(args.modified_output_dir), exist_ok=True)
    
    # Run the comparison
    comparison = SegmentationComparison(args)
    success = comparison.run_comparison()
    
    # Print summary
    print("\n=== Comparison Summary ===")
    if success:
        print("Segmentation comparison test PASSED!")
        print("The refactored pipeline produces equivalent segmentation results to the modified original.")
        return 0
    else:
        print("Segmentation comparison test FAILED!")
        print("There are differences between the modified original and refactored segmentation outputs.")
        print("Check the log file and report JSON for details.")
        return 1

if __name__ == '__main__':
    sys.exit(main())