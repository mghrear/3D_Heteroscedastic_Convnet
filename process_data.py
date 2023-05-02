# This script processes raw simultation files
# You must select either training or testing mode
# In training mode, the added diffusion is drawn randomly from 160-466um (Min and max expected diffusion in a BEAST TPC)
# In testing mode, the added diffusion is specified
# After diffusion a random rotation is applied and the simulations are mean-centered

import root_pandas as rp
import pandas as pd
import ROOT
from ROOT import TVector3, TRandom, TMath
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import mytools

# Select mode: training or testing
mode = 'training'

# For testing mode, select a specific sigma
sigma_test = 400*1e-4

# Specify location of data file
data_loc = '/Users/majdghrear/data/e_dir_fit'

# The data is stored in 100 root files each containing 10k electron recoil simulations
num_files = 300
files_e = [data_loc+'/raw_'+mode+'_data/he_co2_50keV_'+str(i)+'/he_co2_50keV_'+str(i)+'.root' for i in range(num_files) ]

# Total sigma [cm] used for diffusion
sigma = 466.0 * float(1e-4)


# Loop through files
for ind, file in enumerate(files_e):

	# Read root file
	df = rp.read_root(file)

	# Create dataframe for new data
	df2 = pd.DataFrame(columns = ['x', 'y', 'z', 'dir','offset','diff'])

	# Loop through the electron recoil simulations and process them
	for index, row in df.iterrows():

		if mode == 'training':
			# Total sigma [cm] used for diffusion, drawn from uniform distribution between 160um and 466um
			sigma = np.random.uniform(160*1e-4,466*1e-4)

		else:
			sigma = sigma_test

    	# diffuse x/y/z positions
		x_diff = row['x']+(sigma*np.random.normal(size=len(row['x'])))
		y_diff = row['y']+(sigma*np.random.normal(size=len(row['y'])))
		z_diff = row['z']+(sigma*np.random.normal(size=len(row['z'])))

		# Determine random direction to rotate to
		to_dir, theta, phi = mytools.random_three_vector()

		# Rotate all points to random direction drawn in previous step
		Rcoords = mytools.R_z(phi)@mytools.R_y(theta)@np.array((x_diff,y_diff,z_diff))

		x_rot = Rcoords[0]
		y_rot = Rcoords[1]
		z_rot = Rcoords[2]

		# Find the mean position
		mean_x = np.mean(x_rot)
		mean_y = np.mean(y_rot)
		mean_z = np.mean(z_rot)

		# change to mean-centered coordinates, this must be done after rotation and diffusion
		x_final = x_rot-mean_x # This notation automatically promotes x_new to a numpy array (so no error)
		y_final = y_rot-mean_y
		z_final = z_rot-mean_z

		# Store transformed positions and new dataframe
		df2 = df2.append({'x' : x_final, 'y' : y_final, 'z' : z_final, 'dir' : to_dir, 'offset' :  -1.0*np.array( [mean_x, mean_y, mean_z] ), 'diff' : sigma }, ignore_index = True)

	# Save file
	df2.to_pickle('~/data/e_dir_fit/processed_'+mode+'_data/processed_recoils_'+str(ind)+'.pk')


