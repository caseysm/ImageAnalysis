#%%
import numpy as np
import matplotlib.pyplot as plt
import In_Situ_Functions as isf

#%%
img = np.load('/home/mogilevskc/shalemlab/Personal/Casey/New_Analysis/segmented/10X/cells/well_1/Seg_Cells-Well_1_Tile_34.npy'
              '')

path_name = '/home/mogilevskc/shalemlab/Tutorials/Demo_In-Situ-Seq_Analysis/Data/genotyping/cycle_1'
raw = isf.InSitu.Import_ND2_by_Tile_and_Well(34, 1, path_name)
#%%
plt.subplot(1, 2, 1)
plt.imshow(raw[0])

plt.subplot(1, 2, 2)
plt.imshow(img[0])

plt.show()
#%%
path_name = '/mnt/isilon/shalemlab/Tutorials/Demo_In-Situ-Seq_Analysis/Data/phenotyping'
raw = isf.InSitu.Import_ND2_by_Tile_and_Well(1039, 1, path_name)
nucs = isf.Segment.Segment_Nuclei(raw[0], nuc_diameter=120)
plt.imshow(nucs)
plt.show()
#%%
from cellpose import models as cellpose_models
nuclei_model = cellpose_models.Cellpose(model_type='cyto3')
masks, _, _, _ = nuclei_model.eval(raw[0], diameter=120, channels=[[0, 0]])
plt.imshow(masks)
plt.show()