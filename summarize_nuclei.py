#\!/usr/bin/env python3
import json
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Find all metadata files
base_dir = "results/segmentation_output"
metadata_files = glob.glob(f"{base_dir}/**/*_metadata.json", recursive=True)

data = []
for file_path in metadata_files:
    with open(file_path, 'r') as f:
        meta = json.load(f)
        image_file = Path(meta['image_file']).name
        magnification = "10X" if "10X" in file_path else "40X"
        data.append({
            'image_file': image_file,
            'magnification': magnification,
            'num_nuclei': meta['num_nuclei']
        })

# Create DataFrame
df = pd.DataFrame(data)

# Summary statistics by magnification
summary = df.groupby('magnification')['num_nuclei'].agg([
    'count', 'mean', 'std', 'min', 'max'
]).reset_index()

# Print results
print("\nNuclei Count Summary by Magnification:")
print(summary.to_string(index=False))

# Print all nuclei counts
print("\nDetailed Nuclei Counts:")
for mag in ['10X', '40X']:
    print(f"\n{mag} Images:")
    subset = df[df['magnification'] == mag]
    for _, row in subset.iterrows():
        print(f"  {row['image_file']}: {row['num_nuclei']} nuclei")

# Visualize nuclei counts
plt.figure(figsize=(10, 6))
plt.bar(['10X', '40X'], 
        [df[df['magnification'] == '10X']['num_nuclei'].mean(), 
         df[df['magnification'] == '40X']['num_nuclei'].mean()],
        yerr=[df[df['magnification'] == '10X']['num_nuclei'].std(), 
              df[df['magnification'] == '40X']['num_nuclei'].std()],
        alpha=0.7, capsize=10)
plt.ylabel('Average Number of Nuclei')
plt.title('Average Number of Nuclei by Magnification')
plt.grid(axis='y', alpha=0.3)
plt.savefig('results/nuclei_count_comparison.png')

# Save detailed CSV
df.to_csv('results/nuclei_counts.csv', index=False)
