import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch



# Plot a charge distribution as well as the initial direction (label)
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


# Plot a tensor as well as the initial direction (label)
def plot_tensor_dir(tensor, start, direction, eff_l, vox_l):

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

    x = np.sin( theta) * np.cos( phi )
    y = sp.sin( theta) * np.sin( phi )
    z = np.cos( theta )

    return np.array([x,y,z])

# Class for creating pytorch DataSet
class CustomDataset(torch.utils.data.Dataset):
    
    def __init__(self,dir_loc, N_sims):
        self.dir_loc = dir_loc
        self.N_sims = N_sims
    
    def __len__(self):
        return self.N_sims
    
    def __getitem__(self,idx):
        return ( torch.load(self.dir_loc + 'sparse_recoils_' + str(idx) + '.pt' ), torch.load(self.dir_loc + 'label_' + str(idx) + '.pt' ), idx )
    
    

# Define training epoch loop
def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss = 0
    for batch, (X, y, index) in enumerate(dataloader):
        
        X, y = X.type(torch.FloatTensor).to(device), y.to(device)
        
        #convert to dense tensor
        X = X.to_dense()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()        

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Current batch training loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    train_loss /= num_batches
    print(f"Training loss: {train_loss:>7f}")
    return(train_loss)


# Define validation epoch loop
def validate(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, y, index in dataloader:
            X, y = X.type(torch.FloatTensor).to(device), y.to(device)
            
            #convert to dense tensor
            X = X.to_dense()
                        
            pred = model(X)

            val_loss += loss_fn(pred, y).item()
            
    val_loss /= num_batches
    print(f"Validation loss: {val_loss:>7f} \n")
    return(val_loss)



