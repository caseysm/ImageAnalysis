import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from natsort import natsorted
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import os, sys

# Concatenate genotyped cells list, 'Cells_Genotyped_Complete_Well_x.csv', from every well
# into a single complete genotyped cell list

# Get a list of all genotyped cells files
path = "genotyping_results/"
wells = [d for d in listdir(path) if d.startswith('well_')]
print(f"Found wells: {wells}")

# Concatenate all well files
df_list = []
for well_dir in tqdm(wells):
    well_path = join(path, well_dir)
    try:
        file_name = [f for f in listdir(well_path) if f.startswith('Cells_Genotyped_Complete')][0]
        df_well = pd.read_csv(join(well_path, file_name))
        df_list.append(df_well)
        print(f"{file_name}: {len(df_well)} cells")
    except:
        print(f"Could not process well: {well_dir}")

# Combine all dataframes into one
df_all = pd.concat(df_list, ignore_index=True)
print(f"Total genotyped cells: {len(df_all)}")

# Save the combined dataframe
df_all.to_csv("Cells_Genotyped.csv", index=False)
print("Saved combined genotype data to Cells_Genotyped.csv")

# Check for single cell image dataframes
single_cell_path = "datasets/single_cell_dataframes"
if os.path.exists(single_cell_path):
    dataframe_files = natsorted([f for f in listdir(single_cell_path) 
                                if isfile(join(single_cell_path, f)) and f.endswith('.csv')])
    
    # Concatenate single cell dataframes
    df_single_cells = []
    for file in tqdm(dataframe_files):
        df_batch = pd.read_csv(join(single_cell_path, file))
        df_single_cells.append(df_batch)
    
    df_all_single_cells = pd.concat(df_single_cells, ignore_index=True)
    print(f"Total cells with images: {len(df_all_single_cells)}")
    
    # Save the combined single cell dataframe
    df_all_single_cells.to_csv("Cells_Imaged.csv", index=False)
    print("Saved combined cell image data to Cells_Imaged.csv")
    
    # Optional: Display a sample cell
    sample_cells = df_all_single_cells[df_all_single_cells['image'].notna()]
    if len(sample_cells) > 0:
        img_path = sample_cells['image'].iloc[0]
        print(f"Sample cell image: {img_path}")
        
        img = np.array(Image.open('datasets/single_cells/' + img_path))
        cell_mask = img[:, :, -1] > 150
        nuc_mask = img[:, :, -1] > 200
        
        plt.figure(figsize=(10, 6))
        for ch in range(4):
            plt.subplot(1, 4, ch + 1)
            plt.imshow(img[:, :, ch]*cell_mask)
            plt.axis('off')
        plt.show()

print("Concatenation process complete.") 