import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from PIL import Image
import nd2reader
import os
import sys
import warnings
import natsort
from os import listdir
from os.path import isfile, join

# Add parent directory to import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from modified pipeline
from modified_original_pipeline import In_Situ_Functions as isf

def Num_to_Str(_i):

    # Convert integer number to four digit string, as used by nikon nd2 file notation
    # Input integer, output string

    if _i < 10:
        out = '000' + str(_i)
    if _i >= 10 and _i < 100:
        out = '00' + str(_i)
    if _i >= 100 and _i < 1000:
        out = '0' + str(_i)
    if _i > 1000:
        out = str(_i)

    return out

def Open_nd2_to_npy(_path):
    return np.array(nd2reader.ND2Reader(_path), dtype=np.float64)

def Plot_4_Center_Tiles(_M, _path_name, well=1, channel=0, shift=0):

    # Stitch 4 adjacent tiles into a single image
    # _M: tile location matrix
    # _path_name: path to directory containing of high amg images
    # well: well of interest
    # channel: channel number for plot, deflaut is 0, typically nuclear DAPI channel
    # shift: placement in tile location matrix, default is the center tile. shift will move integer number of tiles along i and j axes

    H_M, W_M = _M.shape
    I = int(H_M/2) + shift
    J = int(W_M/2) + shift

    T = np.zeros([4], dtype=int)
    T[0] = _M[I, J]
    T[1] = _M[I + 1, J]
    T[2] = _M[I, J + 1]
    T[3] = _M[I + 1, J + 1]

    img_test = isf.InSitu.Import_ND2_by_Tile_and_Well(T[0], well, _path_name)

    assert img_test.ndim == 3, 'Image must be 3 dimension: Channel, Height, Width'
    _, R, W = img_test.shape
    assert R == W, 'Image must be square'

    img_stack = np.empty([4, R, R])
    for t in range(4):
        img_stack[t] = isf.InSitu.Import_ND2_by_Tile_and_Well(T[t], well, _path_name)[channel]

    master = np.zeros([R * 2, R * 2])

    master[:R, :R] = img_stack[0]
    master[:R, R:] = img_stack[2]
    master[R:, :R] = img_stack[1]
    master[R:, R:] = img_stack[3]

    fig = plt.figure()
    ax = fig.subplots(1)
    fig.set_size_inches(20, 20)
    ax.imshow(master)

def Get_Image_Size(_path_name, verbose=False):

    # Get H and W of nd2 image from directory of nd2 files. Using the first file in directory

    onlyfiles = [f for f in listdir(_path_name) if isfile(join(_path_name, f)) and join(_path_name, f).endswith('.nd2')]

    with nd2reader.ND2Reader(join(_path_name, onlyfiles[0])) as nd2_file:
        _h = nd2_file.metadata['height']
        _w = nd2_file.metadata['width']
        _shape = nd2_file.shape

    if verbose:
        print('Height:', _h, 'Width:', _w)
        print('Image shape:', _shape)

    return _h, _w

def Local_to_Global(_P, _M, _Size):

    # Convert local position of point in tile to a global well coordinate based on tile location in well
    # _P: points, 2D numpy array of size (number of points x 3), array columns are: tile number, i coordinate, j coordinate
    # _M: tile location matrix
    # _Size: numpy array of structure [H, W], of the H and W of tile
    # return: 2D numpy array of size (number of points x 3), array columns are: tile number, i coordinate, j coordinate

    P_g = np.zeros([len(_P), 3])
    P_g[:, 0] = _P[:, 0]

    h, w = _Size

    # Find in which i,j indices each point is in _M
    M_indices = np.empty([len(_P), 2], dtype=int)
    for i, p in enumerate(_P):
        tile = int(p[0])
        M_indices[i, :] = np.where(_M == tile)

    # i global = ((i from _M) * H) + (i local from _P)
    # j global = ((j from _M) * W) + (j local from _P)
    P_g[:, 1] = M_indices[:, 0] * h + _P[:, 1]
    P_g[:, 2] = M_indices[:, 1] * w + _P[:, 2]

    return P_g

def Global_to_Local(_P_g, _M, _Size):

    # Convert global position of point in well to a local tile coordinate
    # _P_g: points, 2D numpy array of size (number of points x 3), array columns are: tile number, i coordinate, j coordinate
    # _M: tile location matrix
    # _Size: numpy array of structure [H, W], of the H and W of tile
    # return: 2D numpy array of size (number of points x 3), array columns are: tile number, i coordinate, j coordinate

    P_local = np.zeros([len(_P_g), 3])
    P_local[:, 0] = _P_g[:, 0]

    h, w = _Size

    # Find in which i,j indices each point is in _M
    M_indices = np.empty([len(_P_g), 2], dtype=int)
    for i, p in enumerate(_P_g):
        tile = int(p[0])
        M_indices[i, :] = np.where(_M == tile)

    # i local = i global - ((i from _M) * H)
    # j local = j global - ((j from _M) * W)
    P_local[:, 1] = _P_g[:, 1] - (M_indices[:, 0] * h)
    P_local[:, 2] = _P_g[:, 2] - (M_indices[:, 1] * w)

    return P_local

def Invert_M_Matrix(_M):

    # Compute how many cells are in each tile in the tile location matrix _M in a spiral pattern
    # _M: tile location matrix
    # return: matrix of cells per tile, with same dimensions as _M

    _max = np.max(_M)

    # Initialize a matrix of zeros with same shape as _M
    _count = np.zeros_like(_M)
    
    # Spiral pattern to fill _count
    for i in range(1, _max + 1):
        idx = np.where(_M == i)
        if len(idx[0]) > 0:
            _count[idx] = i

    return _count

def Map_10X_to_40X(_L_10X, _M_10X, _M_40X, _Size_10X, _Size_40X, DOF=6, verbose=False):

    # Map centroid coordinates from 10X to 40X magnification
    # _L_10X: 2D numpy array with columns [tile, i, j] of 10X centroid coordinates
    # _M_10X: tile location matrix for 10X
    # _M_40X: tile location matrix for 40X
    # _Size_10X: numpy array [H, W] of 10X tile size
    # _Size_40X: numpy array [H, W] of 40X tile size
    # DOF: degrees of freedom for transformation (6 or 8)
    # return: 2D numpy array with columns [tile, i, j] of mapped 40X centroid coordinates

    # Convert local 10X coordinates to global
    _G_10X = Local_to_Global(_L_10X, _M_10X, _Size_10X)

    # Scale from 10X to 40X
    scale_factor = 4  # 40X / 10X
    _G_40X = _G_10X.copy()
    _G_40X[:, 1:] = _G_10X[:, 1:] * scale_factor

    # Map to appropriate 40X tiles
    h_40X, w_40X = _Size_40X
    h_M, w_M = _M_40X.shape

    # Initialize output array
    _L_40X = np.zeros_like(_L_10X)

    # For each point, find which 40X tile it falls in and convert to local coordinates
    for i in range(len(_G_40X)):
        i_global = _G_40X[i, 1]
        j_global = _G_40X[i, 2]

        # Find which tile this point belongs to
        i_tile = int(i_global // h_40X)
        j_tile = int(j_global // w_40X)
        
        # Check if coordinates are within matrix bounds
        if i_tile < h_M and j_tile < w_M and i_tile >= 0 and j_tile >= 0:
            tile_40X = _M_40X[i_tile, j_tile]
            
            # Calculate local coordinates
            i_local = i_global % h_40X
            j_local = j_global % w_40X
            
            _L_40X[i, 0] = tile_40X
            _L_40X[i, 1] = i_local
            _L_40X[i, 2] = j_local
        else:
            # Point is outside the bounds of available tiles
            _L_40X[i, 0] = -1  # Mark as invalid
            _L_40X[i, 1] = -1
            _L_40X[i, 2] = -1

    if verbose:
        # Calculate how many points were successfully mapped
        valid_points = np.sum(_L_40X[:, 0] != -1)
        print(f"Mapped {valid_points} out of {len(_L_10X)} points ({valid_points/len(_L_10X)*100:.2f}%)")

    return _L_40X

def Map_40X_to_10X(_L_40X, _M_10X, _M_40X, _Size_10X, _Size_40X, DOF=6, verbose=False):

    # Map centroid coordinates from 40X to 10X magnification
    # _L_40X: 2D numpy array with columns [tile, i, j] of 40X centroid coordinates
    # _M_10X: tile location matrix for 10X
    # _M_40X: tile location matrix for 40X
    # _Size_10X: numpy array [H, W] of 10X tile size
    # _Size_40X: numpy array [H, W] of 40X tile size
    # DOF: degrees of freedom for transformation (6 or 8)
    # return: 2D numpy array with columns [tile, i, j] of mapped 10X centroid coordinates

    # Convert local 40X coordinates to global
    _G_40X = Local_to_Global(_L_40X, _M_40X, _Size_40X)

    # Scale from 40X to 10X
    scale_factor = 0.25  # 10X / 40X
    _G_10X = _G_40X.copy()
    _G_10X[:, 1:] = _G_40X[:, 1:] * scale_factor

    # Map to appropriate 10X tiles
    h_10X, w_10X = _Size_10X
    h_M, w_M = _M_10X.shape

    # Initialize output array
    _L_10X = np.zeros_like(_L_40X)

    # For each point, find which 10X tile it falls in and convert to local coordinates
    for i in range(len(_G_10X)):
        i_global = _G_10X[i, 1]
        j_global = _G_10X[i, 2]

        # Find which tile this point belongs to
        i_tile = int(i_global // h_10X)
        j_tile = int(j_global // w_10X)
        
        # Check if coordinates are within matrix bounds
        if i_tile < h_M and j_tile < w_M and i_tile >= 0 and j_tile >= 0:
            tile_10X = _M_10X[i_tile, j_tile]
            
            # Calculate local coordinates
            i_local = i_global % h_10X
            j_local = j_global % w_10X
            
            _L_10X[i, 0] = tile_10X
            _L_10X[i, 1] = i_local
            _L_10X[i, 2] = j_local
        else:
            # Point is outside the bounds of available tiles
            _L_10X[i, 0] = -1  # Mark as invalid
            _L_10X[i, 1] = -1
            _L_10X[i, 2] = -1

    if verbose:
        # Calculate how many points were successfully mapped
        valid_points = np.sum(_L_10X[:, 0] != -1)
        print(f"Mapped {valid_points} out of {len(_L_40X)} points ({valid_points/len(_L_40X)*100:.2f}%)")

    return _L_10X

def Optimize_Mapping(_L_10X, _L_40X, DOF=6, iterations=10000, verbose=False):
    
    # Optimize the mapping between 10X and 40X coordinates
    # _L_10X: 2D numpy array with columns [tile, i, j] of 10X centroid coordinates
    # _L_40X: 2D numpy array with columns [tile, i, j] of 40X centroid coordinates
    # DOF: degrees of freedom for transformation (6 or 8)
    # iterations: number of optimization iterations
    # return: optimized transformation parameters

    # Filter out invalid points
    valid_mask = (_L_10X[:, 0] != -1) & (_L_40X[:, 0] != -1)
    _L_10X_valid = _L_10X[valid_mask, 1:]  # Use only i,j coordinates
    _L_40X_valid = _L_40X[valid_mask, 1:]  # Use only i,j coordinates
    
    if len(_L_10X_valid) == 0:
        if verbose:
            print("No valid points found for optimization")
        return None

    if verbose:
        print(f"Optimizing mapping with {len(_L_10X_valid)} valid points")

    # Define the transformation function based on DOF
    def transform(params, points):
        if DOF == 6:
            # Affine transformation: [scale_x, scale_y, rotation, shear, tx, ty]
            scale_x, scale_y, theta, shear, tx, ty = params
            
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            
            transformed = np.zeros_like(points)
            for i, (x, y) in enumerate(points):
                transformed[i, 0] = scale_x * (x * cos_theta - y * sin_theta) + tx
                transformed[i, 1] = scale_y * (x * sin_theta + y * cos_theta + shear * x) + ty
                
            return transformed
        elif DOF == 8:
            # Perspective transformation: [a, b, c, d, e, f, g, h]
            a, b, c, d, e, f, g, h = params
            
            transformed = np.zeros_like(points)
            for i, (x, y) in enumerate(points):
                denominator = g * x + h * y + 1
                transformed[i, 0] = (a * x + b * y + c) / denominator
                transformed[i, 1] = (d * x + e * y + f) / denominator
                
            return transformed
        else:
            raise ValueError("DOF must be 6 or 8")

    # Define the cost function
    def cost_function(params):
        transformed = transform(params, _L_10X_valid)
        return np.mean(np.sqrt(np.sum((_L_40X_valid - transformed) ** 2, axis=1)))

    # Initial parameter guess
    if DOF == 6:
        # [scale_x, scale_y, rotation, shear, tx, ty]
        initial_params = [4.0, 4.0, 0.0, 0.0, 0.0, 0.0]
    else:
        # [a, b, c, d, e, f, g, h]
        initial_params = [4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0]

    # Run optimization
    result = minimize(cost_function, initial_params, method='Nelder-Mead', 
                      options={'maxiter': iterations, 'disp': verbose})

    if verbose:
        print(f"Optimization completed: {result.success}")
        print(f"Final error: {result.fun}")
        print(f"Optimized parameters: {result.x}")

    return result.x

def Apply_Transformation(_L_10X, transformation_params, DOF=6):
    
    # Apply transformation parameters to 10X coordinates to get 40X coordinates
    # _L_10X: 2D numpy array with columns [tile, i, j] of 10X centroid coordinates
    # transformation_params: parameters from Optimize_Mapping
    # DOF: degrees of freedom for transformation (6 or 8)
    # return: transformed coordinates

    # Create a copy to preserve tile numbers
    transformed = _L_10X.copy()
    
    # Transform only valid points
    valid_mask = _L_10X[:, 0] != -1
    points = _L_10X[valid_mask, 1:]  # Use only i,j coordinates
    
    if DOF == 6:
        # Affine transformation
        scale_x, scale_y, theta, shear, tx, ty = transformation_params
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        for i, (x, y) in enumerate(points):
            idx = np.where(valid_mask)[0][i]
            transformed[idx, 1] = scale_x * (x * cos_theta - y * sin_theta) + tx
            transformed[idx, 2] = scale_y * (x * sin_theta + y * cos_theta + shear * x) + ty
            
    elif DOF == 8:
        # Perspective transformation
        a, b, c, d, e, f, g, h = transformation_params
        
        for i, (x, y) in enumerate(points):
            idx = np.where(valid_mask)[0][i]
            denominator = g * x + h * y + 1
            transformed[idx, 1] = (a * x + b * y + c) / denominator
            transformed[idx, 2] = (d * x + e * y + f) / denominator
            
    else:
        raise ValueError("DOF must be 6 or 8")
        
    return transformed