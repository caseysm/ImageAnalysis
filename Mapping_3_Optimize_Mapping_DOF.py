import Mapping_Functions as mf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from pims_nd2 import ND2_Reader as nd2_opener

# Define well for analysis
well = 1

# Load fiducial points
p_10X = np.load(f'fiducial_points_10X_well_{well}.npy')
p_40X = np.load(f'fiducial_points_40X_well_{well}.npy')

print(f"Loaded fiducial points for well {well}")
print("10X points:", p_10X)
print("40X points:", p_40X)

# Load tile matrices
M_10X = np.load('M_10X.npy')
M_40X = np.load('M_40X.npy')

# Convert local coordinates to global coordinates
P_10X = mf.Local_to_Global(p_10X, M_10X, [2304, 2304])
P_40X = mf.Local_to_Global(p_40X, M_40X, [2304, 2304])

# Fit transformation parameters using the fiducial points
DOF = mf.Fit_By_Points(P_10X, P_40X, verbose=True)

print("Optimized transformation parameters (DOF):", DOF)

# Save the transformation parameters
np.save(f'transformation_DOF_well_{well}.npy', DOF)
print(f"Transformation parameters for well {well} saved successfully.") 