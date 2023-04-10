import numpy as np
from ROOT import TVector3, TRandom, TMath
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def plot_track_dir(x_points, y_points, z_points, start, direction):

    # Plot the track
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x_points, y_points, z_points, c='k', marker='o')

    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    ax.set_zlabel('z [cm]')
    plt.tight_layout()

    # Add red line for true direction (z direction)
    #ax.plot3D([vector_tail[0],vector_head[0]], [vector_tail[1],vector_head[1]], [vector_tail[2],vector_head[2]], c='r')
    ax.quiver(start[0],start[1],start[2],direction[0],direction[1],direction[2], linewidths=3, color = 'red')

    plt.show()



def plot_tensor(tensor, start, direction, eff_l, vox_l):

    direction = direction/vox_l
    start = (start + eff_l)/vox_l

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')

    ax.voxels(tensor[0,:,:,:])
    ax.quiver(start[0],start[1],start[2],direction[0],direction[1],direction[2], linewidths=3, color = 'red')

    plt.show()






# This function draws an a 3-D vector from an isotropic distribution
def random_three_vector():



    phi = np.random.uniform()*2*np.pi

    costheta = 2.0*np.random.uniform()-1.0
    theta = TMath.ACos( costheta )

    x = TMath.Sin( theta) * TMath.Cos( phi )
    y = TMath.Sin( theta) * TMath.Sin( phi )
    z = TMath.Cos( theta )

    return TVector3(x,y,z)

