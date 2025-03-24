import Mapping_Functions as mf
import In_Situ_Functions as isf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pims_nd2 import ND2_Reader as nd2_opener
from os.path import join
import sys

# Define paths to image directories
path_10X = '/mnt/isilon/shalemlab/Tutorials/Demo_In-Situ-Seq_Analysis/Data/genotyping/cycle_1'
path_40X = '/mnt/isilon/shalemlab/Tutorials/Demo_In-Situ-Seq_Analysis/Data/phenotyping'

# Load tile matrices
M_10X = np.load('M_10X.npy')
M_40X = np.load('M_40X.npy')

# Define well for analysis
well = 1

# Manual function to identify fiducial cells
def find_fiducial_cell(well, T_10X, i_10X, j_10X, T_40X, i_40X, j_40X):
    # Display 10X image with selected cell
    img_10X = isf.InSitu.Import_ND2_by_Tile_and_Well(T_10X, well, path_10X)
    plt.figure(figsize=(6, 6), dpi=80)
    plt.imshow(img_10X[0])
    plt.scatter([j_10X], [i_10X], s=3, c='red')
    plt.title(f"10X Well {well}, Tile {T_10X}")
    plt.show()
    
    # Display 40X image with selected cell
    img_40X = isf.InSitu.Import_ND2_by_Tile_and_Well(T_40X, well, path_40X)
    plt.figure(figsize=(6, 6), dpi=80)
    plt.imshow(img_40X[0])
    plt.scatter([j_40X], [i_40X], s=3, c='red')
    plt.title(f"40X Well {well}, Tile {T_40X}")
    plt.show()
    
    return np.array([T_10X, i_10X, j_10X]), np.array([T_40X, i_40X, j_40X])

# Example: Find multiple fiducial points for well 1
# The user should modify these values manually for their specific images
# Point 1
print("Identifying fiducial point 1:")
p_10X_1, p_40X_1 = find_fiducial_cell(well, 34, 1610, 228, 561, 2030, 1285)

# Point 2
print("Identifying fiducial point 2:")
p_10X_2, p_40X_2 = find_fiducial_cell(well, 35, 1295, 672, 566, 758, 765)

# Point 3
print("Identifying fiducial point 3:")
p_10X_3, p_40X_3 = find_fiducial_cell(well, 50, 730, 900, 897, 800, 1670)

# Combine points into arrays
p_10X = np.array([p_10X_1, p_10X_2, p_10X_3])
p_40X = np.array([p_40X_1, p_40X_2, p_40X_3])

# Save the fiducial points for next step
np.save(f'fiducial_points_10X_well_{well}.npy', p_10X)
np.save(f'fiducial_points_40X_well_{well}.npy', p_40X)

print(f"Fiducial points for well {well} saved successfully.")
print("10X points shape:", p_10X.shape)
print("40X points shape:", p_40X.shape) 