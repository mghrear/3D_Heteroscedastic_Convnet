import root_pandas as rp
import pandas as pd
import ROOT
from ROOT import TVector3, TRandom, TMath
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import mytools

# Specify location of data file
data_loc = '/Users/majdghrear/data/e_dir_fit'


# The data is stored in 100 root files each containing 10k electron recoil simulations
num_files = 100
files_e = [data_loc+'/raw_data/he_co2_50keV_'+str(i)+'/he_co2_50keV_'+str(i)+'.root' for i in range(num_files) ]


# Total sigma [cm] used for diffusion
sigma = 466.0 * float(1e-4)


# Loop through files
ind=0
for file in files_e:

	# Read root file
	df = rp.read_root(file)

	# Create dataframe for new data
	df2 = pd.DataFrame(columns = ['x', 'y', 'z', 'dir','offset'])

	# Loop through the electron recoil simulations and process them
	for index, row in df.iterrows():

    	# diffuse x/y/z positions
		x_diff = row['x']+(sigma*np.random.normal(size=len(row['x'])))
		y_diff = row['y']+(sigma*np.random.normal(size=len(row['y'])))
		z_diff = row['z']+(sigma*np.random.normal(size=len(row['z'])))

		x_new = []
		y_new = []
		z_new = []

		# Determine random direction to rotate to
		to_dir = mytools.random_three_vector()

		for x,y,z in zip(x_diff,y_diff,z_diff):

			# Extract vector for charge position
			charge = TVector3(x,y,z)

			# Rotate the charge
			charge.RotateY(to_dir.Theta())
			charge.RotateZ(to_dir.Phi())

			x_new += [charge[0]]
			y_new += [charge[1]]
			z_new += [charge[2]]

		mean_x = np.mean(x_new)
		mean_y = np.mean(y_new)
		mean_z = np.mean(z_new)

		# change to mean-centered coordinates, this must be done after rotation and diffusion
		x_final = x_new-mean_x # This notation automatically promotes x_new to a numpy array (so no error)
		y_final = y_new-mean_y
		z_final = z_new-mean_z

		# Store transformed positions and new dataframe
		df2 = df2.append({'x' : x_final, 'y' : y_final, 'z' : z_final, 'dir' : np.array([to_dir[0],to_dir[1],to_dir[2]]), 'offset' :  -1.0*np.array( [mean_x, mean_y, mean_z] ) }, ignore_index = True)

	# Save file
	df2.to_pickle('~/data/e_dir_fit/processed_data/processed_recoils_'+str(ind)+'.pk')
	ind += 1


