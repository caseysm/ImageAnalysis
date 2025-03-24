import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import os, sys
from Assemble_By_Gaussian_Functions import *

# Install required packages if needed
# Uncomment to run installation
# import subprocess
# subprocess.run(["pip", "install", "scikit-learn"])

# Load the dataset
df = pd.read_csv('Cells_Imaged.csv')
print('Number of cells:', len(df))

# Create album directory if it doesn't exist
if not os.path.exists('albums'):
    os.makedirs('albums')

def Album(df_album, ch=1):
    """
    Generate an album of single cell images arranged in a grid
    
    Parameters:
    df_album: DataFrame containing cell information
    ch: Channel to visualize (0=DAPI, 1=G3BP1, 2=RBP-Halo)
    """
    for c, file in enumerate(df_album['image']):
        if c < 50:  # Limit to 50 cells per album
            img = np.moveaxis(np.array(Image.open(os.path.join('datasets/single_cells', file)).resize((128, 128)), dtype=float), -1, 0)
            mask = img[3] > 0
            img_final = img[ch] * mask
            plt.subplot(5, 10, c + 1)
            plt.imshow(img_final, cmap='Greys_r')
            plt.axis('off')

# List of sgRNAs to create albums for
sgRNA_list = ['ADAR_9', 'EIF4G1_7', 'DDX5_13', 'EWSR1_5', 'NOLC1_7',
              'RBFOX2_12', 'UPF1_8', 'HNRNPA1_4', 'IGF2BP1_9', 'KHDRBS1_4']

# For each sgRNA, create an album
for sgRNA in tqdm(sgRNA_list):
    df_sgRNA = df[df['sgRNA'] == sgRNA]
    print(f"Processing {sgRNA}: {len(df_sgRNA)} cells")
    
    # Filter cells by intensity range if needed
    if 'ints_avg_Halo' in df.columns:
        # Get intensity range parameters
        pick_range = find_gaussian_peak(np.log10(df_sgRNA['ints_avg_Halo'].to_numpy()), 
                                        bins=20, h_lim=0.7, verbose=False)[0]
        
        # Filter cells by intensity range
        df_album = df_sgRNA.copy()
        df_album = df_album[df_album['ints_avg_Halo'] >= 10**pick_range[0]]
        df_album = df_album[df_album['ints_avg_Halo'] <= 10**pick_range[1]]
    else:
        df_album = df_sgRNA.copy()
    
    # Create the visualization
    fig = plt.figure(figsize=(12, 7), dpi=150)
    plt.suptitle(sgRNA)
    Album(df_album, ch=1)
    
    # Save the album
    plt.savefig(f'albums/{sgRNA}.png')
    plt.clf()
    plt.close()

print("Album generation complete.") 