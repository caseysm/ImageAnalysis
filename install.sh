#!/bin/bash
# Installation script for imageanalysis package

# Create and activate conda environment
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

# Activate the environment (this requires source to be used)
echo "To activate the environment, run:"
echo "conda activate ImageAnalysis"
echo ""
echo "Then, to install the package in development mode, run:"
echo "pip install -e ."
echo ""
echo "You can then use the package with 'import imageanalysis' in Python"
echo "or run the command-line tools directly."