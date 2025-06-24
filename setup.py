#!/usr/bin/env python3
"""Setup script for the imageanalysis package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="imageanalysis",
    version="0.2.0",
    description="Image analysis package for microscopy pipeline processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Shalem Lab",
    author_email="info@shalemlab.com",
    url="https://github.com/caseysm/ImageAnalysis",
    license="MIT",
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
    extras_require={
        'test': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'pytest-xdist>=3.0.0',
            'pytest-mock>=3.10.0',
            'pytest-timeout>=2.1.0',
            'hypothesis>=6.0.0',
        ],
        'dev': [
            'black>=22.0.0',
            'flake8>=5.0.0',
            'mypy>=0.990',
            'pre-commit>=2.20.0',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords="microscopy image-analysis segmentation cellpose bioinformatics",
)