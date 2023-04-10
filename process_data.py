# This script is used to read the degrad simulation as root files and perform the following:
# 1. Isotropize the simulations and creat a label for the true recoil direction
# 2. Diffuse the simulations
# 3. mean-center the simulations
# 4. Store the simulations as pickles 

import root_pandas as rp
import pandas as pd
import ROOT
from ROOT import TVector3, TRandom, TMath
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Specify location of data file
data_loc = '/Users/majdghrear/data/e_dir_fit'


# The data is stored in 100 root files each containing 10k electron recoil simulations
num_files = 100
files_e = [data_loc+'/raw_data/he_co2_50keV_'+str(i)+'/he_co2_50keV_'+str(i)+'.root' for i in range(num_files) ]


# Define tools to rotate track randomly

# This function rotates the track to a random direction
def rotate_track(track, to_dir):

    for charge in track:

        charge.RotateY(-(0.5*TMath.Pi()-to_dir.Theta()))
        charge.RotateZ(to_dir.Phi())


# This function draws an a 3-D vector from an isotropic distribution
def random_three_vector():

    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    """

    phi = np.random.uniform()*2*np.pi

    costheta = 2.0*np.random.uniform()-1.0
    theta = TMath.ACos( costheta )

    x = TMath.Sin( theta) * TMath.Cos( phi )
    y = TMath.Sin( theta) * TMath.Sin( phi )
    z = TMath.Cos( theta )

    return TVector3(x,y,z)


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
		# make lists for transformed data
		x_new = []
		y_new = []
		z_new = []

		# Determine random direction to rotate to
		to_dir = random_three_vector()

		for x,y,z in zip(row['x'],row['y'],row['z']):

			# Extract vector for charge position
			charge = TVector3(x,y,z)

			# Rotate the charge
			charge.RotateY(to_dir.Theta())
			charge.RotateZ(to_dir.Phi())

			# Add diffusion
			charge += TVector3(sigma* np.random.normal(),sigma* np.random.normal(),sigma* np.random.normal())

			x_new += [charge[0]]
			y_new += [charge[1]]
			z_new += [charge[2]]

		# change to mean-centered coordinates, this must be done after rotation and diffusion
		x_final = []
		y_final = []
		z_final = []

		# Find data mean
		mean_dir = TVector3(np.mean(x_new),np.mean(y_new),np.mean(z_new))

		for x,y,z in zip(x_new,y_new,z_new):

			charge = TVector3(x,y,z)
			charge -= mean_dir

			x_final += [charge[0]]
			y_final += [charge[1]]
			z_final += [charge[2]]

		# Store transformed positions and new dataframe
		df2 = df2.append({'x' : x_final, 'y' : y_final, 'z' : z_final, 'dir' : [to_dir[0],to_dir[1],to_dir[2]], 'offset' : [mean_dir[0],mean_dir[1],mean_dir[2]] }, ignore_index = True)

	# Save file
	df2.to_pickle('~/data/e_dir_fit/3D_processed_data/processed_recoils_'+str(ind)+'.pk')
	ind += 1


