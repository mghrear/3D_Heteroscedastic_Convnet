# This script is used to read the processed pickle files from "process_data.py" and perform the following:
# 1. Voxelize the data
# 2. store the data as pytorch sparse tensors, to be analyzed by a convolutional neural network

import pandas as pd
import numpy as np
import torch

# Select mode: training or testing
mode = 'training'

# Specify location of input data files
data_loc = '/Users/majdghrear/data/e_dir_fit/processed_'+mode+'_data/'

# Specify output directory
out_dir = '/Users/majdghrear/data/e_dir_fit/sparse_'+mode+'_tensors/'

# The data is stored in 100 pickle files each containing 10k electron recoil simulations
files_e = [data_loc+'processed_recoils_'+str(i)+'.pk' for i in range(300) ]


# Here a define the pixel grid parameters
# x/y/z length being kept in cm
eff_l = 3.0
# Voxel size in cm
vox_l = 0.05
# Number of voxels along 1 dim
Npix = round(eff_l*2/vox_l) 
# Tensor dimensions, there is an extra dimension for color which is not used
dim = (1,Npix,Npix,Npix)

# counter for tracks that are not contained
cnt = 0

# dataframe to store labels, offsets and applied diffusion
df2 = pd.DataFrame(columns = ['dir','offset','diff'])

# Loop through files
for ind, file in enumerate(files_e):

	# Read root file
	df = pd.read_pickle(file)

	# Loop thtough processed simulations in file
	for index, row in df.iterrows():

		# If recoil escapes fiducial area, skip it
		if np.max(row['x']) >= eff_l or np.min(row['x']) < -eff_l or np.max(row['y']) >= eff_l or np.min(row['y']) < -eff_l or np.max(row['z']) >= eff_l or np.min(row['z']) < -eff_l:
			cnt += 1
			continue

		# Initialize empty dense tensor as numpy array
		voxelgrid = np.zeros(dim).astype('uint8')

		# Loop the x, y, z positions in the recoil and fill in the dense tensor
		for x,y,z in zip(row['x'],row['y'],row['z']):
			voxelgrid[0][int((x+eff_l)/vox_l)][int((y+eff_l)/vox_l)][int((z+eff_l)/vox_l)] += 1

		# Convert to pytorch tensor
		voxelgrid = torch.tensor(voxelgrid)
		# Convert to sparse pytorch tensor
		vg = voxelgrid.to_sparse()

		df2 = df2.append({ 'dir' : row['dir'], 'offset' :  row['offset'], 'diff' : row['diff'] }, ignore_index = True)

		# Store sparse tensor and corresponding information
		torch.save( vg, data_loc+'/sparse_tensors/sparse_recoils_'+str(ind)+'.pt')

df2.to_pickle(out_dir+'sparse_tensor_info.pk')
print("tracks not contained: ", cnt)