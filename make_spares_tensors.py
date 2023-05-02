# This script is used to read the processed pickle files from "process_data.py" and perform the following:
# 1. Voxelize the data
# 2. store the data as pytorch sparse tensors, to be analyzed by a convolutional neural network

import pandas as pd
import numpy as np
import torch

# Specify location of data file
data_loc = '/Users/majdghrear/data/e_dir_fit'

# The data is stored in 100 pickle files each containing 10k electron recoil simulations
files_e = [data_loc+'/processed_training_data/processed_recoils_'+str(i)+'.pk' for i in range(300) ]


# Here a define the pixel grid parameters
# x/y/z length being kept in cm
eff_l = 3.0
# Voxel size in cm
vox_l = 0.05
# Number of voxels along 1 dim
Npix = round(eff_l*2/vox_l) 
# Tensor dimensions, there is an extra dimension for color which is not used
dim = (1,Npix,Npix,Npix)


# Loop through files
ind = 0
cnt = 0

# dataframe to store labels, offsets and applied diffusion
df2 = pd.DataFrame(columns = ['dir','offset','diff'])

for file in files_e:

    # Read root file
    df = pd.read_pickle(file)

    # Loop thtough processed simulations in file
    for index, row in df.iterrows():

        # If recoil escapes fiducial area, skip it
        if np.max(row['x']) >= eff_l or np.min(row['x']) < -eff_l or np.max(row['y']) >= eff_l or np.min(row['y']) < -eff_l or np.max(row['z']) >= eff_l or np.min(row['z']) < -eff_l:
            cnt += 1
            print("tracks not contained: ", cnt)
            continue

        # Create sparse tensor
        indices = torch.stack( (torch.zeros(len(row['x'])), (torch.tensor(row['x'])+eff_l)/vox_l  , (torch.tensor(row['y'])+eff_l)/vox_l, (torch.tensor(row['z'])+eff_l)/vox_l ) ).type(torch.int16)
        values = torch.ones(len(row['x'])).type(torch.float)
        vg = torch.sparse_coo_tensor(indices, values, dim)

        # Easy way to sum up duplicate entries in the sparse tensor above
        vg = vg.to_dense().to_sparse()
        
        df2 = df2.append({ 'dir' : row['dir'], 'offset' :  row['offset'], 'diff' : row['diff'] }, ignore_index = True)

        # Store sparse tensor and corresponding information
        torch.save( vg, data_loc+'/sparse_training_tensors/sparse_recoils_'+str(ind)+'.pt')
        ind += 1

df2.to_pickle(data_loc+'/sparse_training_tensors/sparse_tensor_info.pk')
