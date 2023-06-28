# Script containing functions, classes, and dictionaries that are used throughout.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch import nn
import pandas as pd

# Here we define the pixel grid parameters used throughout
voxel_grid = {
  "eff_l": 3.0, # x/y/z length being kept in cm
  "vox_l": 0.05, # cubic voxel length in cm
}

# Plot a point cloud as well as an arrow indicating the initial direction
def plot_track_dir(x_points, y_points, z_points, start, direction):

    # Plot the track
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x_points, y_points, z_points, c='k', marker='o')

    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    ax.set_zlabel('z [cm]')
    plt.tight_layout()

    # Add red line for true direction
    ax.quiver(start[0],start[1],start[2],direction[0],direction[1],direction[2], linewidths=3, color = 'red')

    plt.show()


# Plot a voxel grid as well as an arrow indicating the initial direction
def plot_tensor_dir(tensor, start, direction, eff_l, vox_l):

    direction = direction/vox_l
    start = (start + eff_l)/vox_l

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')

    ax.voxels(tensor[0,:,:,:])
    ax.quiver(start[0],start[1],start[2],direction[0],direction[1],direction[2], linewidths=3, color = 'red')

    plt.show()

# Plot an arrow in 3D
def plot_arrow(x_points, y_points, z_points):

    # Plot the track
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x_points, y_points, z_points, c='k', marker='o')

    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    ax.set_zlabel('z [cm]')
    
    eps = 1e-16
    ax.axes.set_xlim3d(left=-60.-eps, right=60+eps)
    ax.axes.set_ylim3d(bottom=-60.-eps, top=60+eps) 
    ax.axes.set_zlim3d(bottom=-60.-eps, top=60+eps)

    plt.tight_layout()


    plt.show()

# Plot a voxel grid of the arrow
def vox_plot_arrow(tensor, eff_l, vox_l):


    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')

    ax.voxels(tensor[0,:,:,:])

    plt.show()

# Rotation matrices about z and y
R_z = lambda ang: np.array( [[np.cos(ang),-np.sin(ang),0],[np.sin(ang),np.cos(ang),0],[0,0,1]] )
R_y = lambda ang: np.array( [[np.cos(ang),0,np.sin(ang)],[0,1,0],[-np.sin(ang),0,np.cos(ang)]] )

# Draw a 3-D vector from an isotropic distribution
def random_three_vector():

    phi = np.random.uniform()*2*np.pi

    costheta = 2.0*np.random.uniform()-1.0
    theta = np.arccos( costheta )

    x = np.sin( theta) * np.cos( phi )
    y = np.sin( theta) * np.sin( phi )
    z = np.cos( theta )

    return np.array([x,y,z]), theta, phi

# Create custom pytorch Dataset
class CustomDataset(torch.utils.data.Dataset):
    
    def __init__(self,dir_loc, st_info):
        self.dir_loc = dir_loc
        self.st_info = st_info
        self.N_sims = len(st_info)
    
    def __len__(self):
        return self.N_sims
    
    def __getitem__(self,idx):
        return ( torch.load(self.dir_loc + 'sparse_recoils_' + str(idx) + '.pt' ), torch.Tensor(self.st_info.iloc[idx].dir), torch.Tensor(self.st_info.iloc[idx].offset) )
    
#Loss functions
CS = nn.CosineSimilarity()

# Cosine Similarity Loss for regular convnet
def CSloss(output, target):
    loss = torch.mean(-1.0*CS(output,target))
    return loss

# Negative Log Likelihood Loss for HSCDC convnet
def NLLloss(output, target):
    
    # target us the x parameters in the Kent distribution
    G = output[0] # \gamma_1 parameters in Kent distribution
    K = output[1] # \kappa parameter in Kent distribution
    
    # We either use the true NLL loss (with loss1) or a very close approximation (with loss2) depending on whether loss1 = inf
    loss1 = -1.0 * torch.log(torch.div(K,4*torch.pi*torch.sinh(K))).flatten()
    loss2 = -1.0 * ( torch.log(torch.div(K,2*torch.pi)) - K ).flatten()
    
    # Compute negative log likelihood using Kent distribution
    loss = torch.mean( torch.minimum(loss1,loss2) - ( K.flatten() * torch.sum(G*target,dim=1) ) )
    
    return loss


# Training Loop
def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss = 0
    skip_flag = False
    
    for batch, (X, y, offset) in enumerate(dataloader):
        
        X, y = X.type(torch.FloatTensor).to(device), y.to(device)
        
        X = X.coalesce()
        indices = X.indices().permute(1, 0).contiguous().int()
        features = X.values()
            
        # Compute prediction error
        pred = model(features,indices,X.shape[0])
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        # Check for Nans in gradient (this sometimes happens for the heteroscedastic model)
        for name, param in model.named_parameters():
            # Only check parameters requiring grad
            if param.requires_grad:
                if torch.isnan(param.grad).any():
                    print("Warning: nan gradient found. The current loss is: ", loss.item())
                    skip_flag = True
                    break
        
        # Only update weights if there is no nans in gradient
        if skip_flag == False:
            optimizer.step()
        # Otherwise skip this update and reset skip_flag
        else:
            skip_flag = False
            
        train_loss += loss.item()
            
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Current batch training loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    train_loss /= num_batches
    print(f"Training loss: {train_loss:>7f}")
    return(train_loss)


# Validation Loop
def validate(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, y, offset in dataloader:
            X, y = X.type(torch.FloatTensor).to(device), y.to(device)
            
            X = X.coalesce()
            indices = X.indices().permute(1, 0).contiguous().int()
            features = X.values()
            
            pred = model(features, indices, X.shape[0])

            val_loss += loss_fn(pred, y).item()
            
    val_loss /= num_batches
    print(f"Validation loss: {val_loss:>7f} \n")
    return(val_loss)


# Test Loop for heteroscedastic convnet model
def test_HSCDC(dataloader, model, device):
    v_pred = torch.Tensor([])
    K_pred = torch.Tensor([])
    v_true = torch.Tensor([])
    off_true = torch.Tensor([])

    num_batches = len(dataloader)
    model.eval()
    with torch.no_grad():
        for X, y, offset in dataloader:
            X = X.type(torch.FloatTensor).to(device)
            
            X = X.coalesce()
            indices = X.indices().permute(1, 0).contiguous().int()
            features = X.values()
            
            pred = model(features, indices, X.shape[0])

            G = pred[0].to('cpu') # \gamma_1 parameters in Kent distribution
            K = pred[1].to('cpu') # \kappa parameters in Kent distribution
            
            v_pred = torch.cat((v_pred,G), 0)
            K_pred = torch.cat((K_pred,K), 0)
            v_true = torch.cat((v_true,y), 0)
            off_true = torch.cat((off_true,offset), 0)

    return v_pred, K_pred, v_true, off_true

# Test Loop homoscedastic convnet model
def test_CNN(dataloader, model, device):
    v_pred = torch.Tensor([])
    v_true = torch.Tensor([])
    off_true = torch.Tensor([])

    num_batches = len(dataloader)
    model.eval()
    with torch.no_grad():
        for X, y, offset in dataloader:
            X = X.type(torch.FloatTensor).to(device)
            
            X = X.coalesce()
            indices = X.indices().permute(1, 0).contiguous().int()
            features = X.values()
            
            pred = model(features, indices, X.shape[0]).to('cpu')
            
            v_pred = torch.cat((v_pred,pred), 0)
            v_true = torch.cat((v_true,y), 0)
            off_true = torch.cat((off_true,offset), 0)

    return v_pred, v_true, off_true

# Test Loop for non-ML model
def test_NML(dataframe, model, n_sigma_L = 1.5, n_sigma_H = 3, w_o = 0.05, cheat=False):
    
    v_pred, v_true, off_true = [], [], []
    
    for index, row in dataframe.iterrows():
        
        v_p, flag = model(row.positions[0],row.positions[1],row.positions[2],row.charges, row.dir, n_sigma_L, n_sigma_H, w_o, cheat)
        
        if flag == False:
            continue
            
        else:
            v_pred += [v_p]
            v_true += [row.dir]
            off_true += [row.offset]
            
    v_pred, v_true, off_true = np.asarray(v_pred), np.asarray(v_true), np.asarray(off_true)
    
    return v_pred, v_true, off_true

# Test Loop for non-ML2 model
def test_NML2(dataframe, model, eps):
    
    v_pred, v_true, off_true = [], [], []
    
    for index, row in dataframe.iterrows():
        
        v_p = model(row.positions[0],row.positions[1],row.positions[2],row.charges, row.dir, row.offset, eps)

        v_pred += [v_p]
        v_true += [row.dir]
        off_true += [row.offset]
            
    v_pred, v_true, off_true = np.asarray(v_pred), np.asarray(v_true), np.asarray(off_true)
    
    return v_pred, v_true, off_true