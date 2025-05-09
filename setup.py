#!/usr/bin/env python3
"""Setup script for the imageanalysis package."""

from setuptools import setup, find_packages

setup(
    name="imageanalysis",
    version="0.2.0",
    description="Image analysis package for pipeline processing",
    author="Shalem Lab",
    author_email="info@shalemlab.com",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-image",
        "scipy",
        "nd2reader",
        "cellpose"
    ],
    entry_points={
        'console_scripts': [
            'run_segmentation=imageanalysis.bin.run_segmentation:main',
            'run_mapping=imageanalysis.bin.run_mapping:main',
            'run_genotyping=imageanalysis.bin.run_genotyping:main',
            'run_phenotyping=imageanalysis.bin.run_phenotyping:main',
            'create_albums=imageanalysis.bin.create_albums:main',
        ],
    },
)