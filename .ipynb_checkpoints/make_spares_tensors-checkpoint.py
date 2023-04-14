# This script is used to read the processed pickle files from "process_data.py" and perform the following:
# 1. Voxelize the data
# 2. store the data as pytorch sparse tensors, to be analyzed by a convolutional neural network

import pandas as pd
import numpy as np
import torch

# Specify location of data file
data_loc = '/Users/majdghrear/data/e_dir_fit'

# The data is stored in 100 pickle files each containing 10k electron recoil simulations
files_e = [data_loc+'/processed_data/processed_recoils_'+str(i)+'.pk' for i in range(100) ]


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

		# Initialize empty dense tensor as numpy array
		voxelgrid = np.zeros(dim).astype('uint8')

		# Loop the x, y, z positions in the recoil and fill in the dense tensor
		for x,y,z in zip(row['x'],row['y'],row['z']):
			voxelgrid[0][int((x+eff_l)/vox_l)][int((y+eff_l)/vox_l)][int((z+eff_l)/vox_l)] += 1

		# Convert to pytorch tensor
		voxelgrid = torch.tensor(voxelgrid)
		# Convert to sparse pytorch tensor
		vg = voxelgrid.to_sparse()

		# Store sparse tensor and corresponding information
		torch.save( vg, data_loc+'/sparse_tensors/sparse_recoils_'+str(ind)+'.pt')
		torch.save( torch.Tensor(row['dir']), data_loc+'/sparse_tensors/label_'+str(ind)+'.pt')
		torch.save( torch.Tensor(row['offset']), data_loc+'/sparse_tensors/offset_'+str(ind)+'.pt')
		ind += 1