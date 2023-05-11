# This script is used to read the processed pickle files from "process_data.py" and convert them into sparse tensors

import pandas as pd
import numpy as np
import torch
import mytools

# Specify location of data file
data_loc = '/Users/majdghrear/data/e_dir_fit'

# number of simulation files per energy (300 files per energy containing 10k simulations each)
# For now I will only use half of the processed data - I can update later if I see signs of overfitting
num_files = 100


# Here a define the pixel grid parameters
# x/y/z length being kept in cm                                                                                                                                           
eff_l= mytools.voxel_grid['eff_l']
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


for energy in np.arange(40,55,5):

    # The data is stored in 100 pickle files each containing 10k electron recoil simulations
    files_e = [data_loc+'/processed_training_data/'+str(energy)+'_keV/processed_recoils_'+str(i)+'.pk' for i in range(num_files) ]

    # file counter
    fcnt = 0

    for file in files_e:

        # dataframe to store labels, offsets, applied diffusion, and energy
        df2 = pd.DataFrame(columns = ['dir','offset','diff', 'energy', 'true_index'])

        #Counter for tracks not contained
        cnt = 0

        # Read root file
        df = pd.read_pickle(file)

        # Loop through processed simulations in file
        for index, row in df.iterrows():

            # Sparse tensor indices
            indices = torch.stack( ( (torch.tensor(row['x'])+eff_l)/vox_l  , (torch.tensor(row['y'])+eff_l)/vox_l, (torch.tensor(row['z'])+eff_l)/vox_l ) ).type(torch.int16)

            # If recoil escapes fiducial area, skip it
            if torch.min(indices) < 0 or torch.max(indices) >= Npix:
                cnt += 1

            else:
                # Sparse Values indices
                values = torch.ones( (len(row['x']), 1) ).type(torch.float)
                vg = torch.sparse_coo_tensor(indices, values, dim)

                # Sum up duplicate entries in the sparse tensor above
                vg = vg.coalesce()
            
                # Add tensor info to new dataframe
                df2 = df2.append({ 'dir' : row['dir'], 'offset' :  row['offset'], 'diff' : row['diff'], 'energy' : energy, 'true_index' : ind}, ignore_index = True)

                # Store sparse tensor and corresponding information
                torch.save( vg, data_loc+'/sparse_training_tensors/sparse_recoils_'+str(ind)+'.pt')
                ind += 1
        
        print("finished: ", file)
        print("tracks not contained: ", cnt)

        df2.to_pickle(data_loc+'/sparse_training_tensors_info/sparse_tensor_info'+str(energy)+'_'+str(fcnt)+'.pk')
        fcnt += 1


# After this I merge all the pickle files into a file names sparse_tensor_info.pk which is stored in /sparse_training_tensors/

