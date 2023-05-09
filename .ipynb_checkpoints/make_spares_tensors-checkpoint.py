# This script is used to read the processed pickle files from "process_data.py" and perform the following:
# 1. Voxelize the data
# 2. store the data as pytorch sparse tensors, to be analyzed by a convolutional neural network

import pandas as pd
import numpy as np
import torch
import mytools

# Specify location of data file
#data_loc = '/Users/majdghrear/data/e_dir_fit'
data_loc = '/mnt/scratch/lustre_01/scratch/majd'

# number of simulation files per energy (300 files per energy containing 10k simulations each)
# For now I will only use half of the processed data - I can update later if I see signs of overfitting
num_files = 150


# Here a define the pixel grid parameters
# x/y/z length being kept in cm
eff_l = mytools.voxel_grid['eff_l']
# Voxel size in cm
vox_l = mytools.voxel_grid['vox_l']
# Number of voxels along 1 dim
Npix = round(eff_l*2/vox_l) 
# Tensor dimensions, there is an extra dimension for color which is not used
# note that I do not use (1,Npix,Npix,Npix). This is because of the spconv layers that I use
# I need to reshape if I want to use this data to make dense tensors!!!!
dim = (Npix,Npix,Npix,1)


# index to keep track of simulation number
ind = 0

# dataframe to store labels, offsets, applied diffusion, and energy
df2 = pd.DataFrame(columns = ['dir','offset','diff', 'energy'])


for energy in np.arange(35,55,5):

    # The data is stored in 100 pickle files each containing 10k electron recoil simulations
    files_e = [data_loc+'/processed_training_data/'+str(energy)+'_keV/processed_recoils_'+str(i)+'.pk' for i in range(num_files) ]

    for file in files_e:

        #Counter for tracks not contained
        cnt = 0

        # Read root file
        df = pd.read_pickle(file)

        # Loop thtough processed simulations in file
        for index, row in df.iterrows():

            # Sparse tensor indices
            indices = torch.stack( ( (torch.tensor(row['x'])+eff_l)/vox_l  , (torch.tensor(row['y'])+eff_l)/vox_l, (torch.tensor(row['z'])+eff_l)/vox_l ) ).type(torch.int16)

            # If recoil escapes fiducial area, skip it
            if torch.min(indices) < 0 or torch.max(indices) >= Npix:
                cnt += 1
                continue
            
            # Sparse Values indices
            values = torch.ones( (len(row['x']), 1) ).type(torch.float)
            vg = torch.sparse_coo_tensor(indices, values, dim)

            # Sum up duplicate entries in the sparse tensor above
            vg = vg.coalesce()
            
            # Add tensor info to new dataframe
            df2 = df2.append({ 'dir' : row['dir'], 'offset' :  row['offset'], 'diff' : row['diff'], 'energy' : energy }, ignore_index = True)

            # Store sparse tensor and corresponding information
            torch.save( vg, data_loc+'/sparse_training_tensors/sparse_recoils_'+str(ind)+'.pt')
            ind += 1
        
        print("finished: ", file)
        print("tracks not contained: ", cnt)


df2.to_pickle(data_loc+'/sparse_training_tensors/sparse_tensor_info.pk')

