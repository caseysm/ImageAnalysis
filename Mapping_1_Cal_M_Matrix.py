import Mapping_Functions as mf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path to nd2 image directories
path_10X = '/mnt/isilon/shalemlab/Tutorials/Demo_In-Situ-Seq_Analysis/Data/genotyping/cycle_1'
path_40X = '/mnt/isilon/shalemlab/Tutorials/Demo_In-Situ-Seq_Analysis/Data/phenotyping'

# Create dataframes containing the coordinates and names of tile in a well 
print('10X Tiles')
df_tiles_10X = mf.Get_Tile_Coordinates(path_10X, '10X', well=1)
print('40X Tiles')
df_tiles_40X = mf.Get_Tile_Coordinates(path_40X, '40X', well=1)

# Generate a matrix of tile indices for each well
print('10X')
M_10X = mf.Generate_Matrix(df_tiles_10X)
print('40X')
M_40X = mf.Generate_Matrix(df_tiles_40X)

# Check for image quality and shifts in the matrices
print('10X QC')
M_10X_rotated = mf.QC_Matrix(M_10X, path_10X, well=1)
print('40X QC')
M_40X_rotated = mf.QC_Matrix(M_40X, path_40X, well=1)

# Save corrected tile layout matrix
np.save('M_10X.npy', M_10X_rotated)
np.save('M_40X.npy', M_40X_rotated)

pd.DataFrame(M_10X_rotated).to_csv('M_10X.csv', index=False)
pd.DataFrame(M_40X_rotated).to_csv('M_40X.csv', index=False)

print("Tile matrices generated and saved successfully.") 