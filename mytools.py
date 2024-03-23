# Script containing functions, classes, and dictionaries that are used throughout.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch import nn
import pandas as pd
#from torch.masked import masked_tensor, as_masked_tensor

# Here we define the pixel grid parameters used throughout
voxel_grid = {
  "eff_l": 3.0, # x/y/z total length in cm
  "vox_l": 0.05,# x/y/z cubic voxel length in cm
}

# Plot a point cloud as well as an arrow indicating the initial direction
def plot_track_dir(x_points, y_points, z_points,  start, direction, xlim = (-1,3), ylim = (-1,3), zlim = (-1,3)):

    # Plot the track
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    

    ax.set_xlabel('x [cm]',labelpad = 20,fontsize=20)
    ax.set_ylabel('y [cm]',labelpad = 20,fontsize=20)
    ax.set_zlabel('z [cm]',labelpad = 20,fontsize=20)
    ax.tick_params(labelsize=18)
    ax.set_box_aspect(None, zoom=0.85)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    plt.tight_layout()

    # Add red line for true direction
    ax.quiver(start[0],start[1],start[2],2*direction[0],2*direction[1],2*direction[2], linewidths=4, color = 'red')
    ax.scatter3D(x_points, y_points, z_points, s = 4, c='k', marker='o', alpha=0.1)


# Plot a voxel grid as well as an arrow indicating the initial direction
def plot_tensor_dir(tensor, start, direction, eff_l, vox_l):

    direction = direction/vox_l
    start = (start + eff_l)/vox_l

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')

    ax.set_xlabel('x',labelpad = 40,fontsize=35)
    ax.set_ylabel('y',labelpad = 40,fontsize=35)
    ax.set_zlabel('z',labelpad = 40,fontsize=35)
    ax.tick_params(labelsize=30)
    ax.tick_params(direction='out', pad=20)
    ax.set_box_aspect(None, zoom=0.85)


    
    plt.tight_layout()

    ax.voxels(tensor[0,:,:,:],alpha=0.3)
    ax.quiver(start[0],start[1],start[2],2*direction[0],2*direction[1],2*direction[2], linewidths=4, color = 'red')
    
    plt.tight_layout()



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



# Plot a voxel grid of the arrow
def vox_plot_arrow(tensor, eff_l, vox_l):


    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')

    ax.set_xlabel('x',labelpad = 20,fontsize=25)
    ax.set_ylabel('y',labelpad = 20,fontsize=25)
    ax.set_zlabel('z',labelpad = 20,fontsize=25)
    ax.tick_params(labelsize=20)
    plt.tight_layout()

    ax.voxels(tensor[0,:,:,:])


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
# Loss1 is the true loss function and loss2 is a very close approximation for high K
# Loss1 is unstable, when loss1 = inf, torch.minimum will select loss2 instead
# However, the gradient of the torch.miniumum is unstable when one argument is inf so this doesn't work
def NLLloss(output, target):
    
    # target us the x parameters in the Kent distribution
    G = output[0] # \gamma_1 parameters in Kent distribution
    K = output[1].flatten() # \kappa parameter in Kent distribution
    
    # We either use the true NLL loss (with loss1) or a very close approximation (with loss2) depending on whether loss1 = inf
    loss1 = -1.0 * torch.log(torch.div(K,4*torch.pi*torch.sinh(K)))
    loss2 = -1.0 * ( torch.log(torch.div(K,2*torch.pi)) - K )
    
    # Compute negative log likelihood using Kent distribution
    loss = torch.mean( torch.minimum(loss1,loss2) - ( K * torch.sum(G*target,dim=1) ) )
    
    return loss


# Negative Log Likelihood Loss for HSCDC convnet
# This implementation uses pytorch masked tensors not functions are implemented with masked tensors so this doesn't work
def NLLloss_masked(output, target):
    
    # target us the x parameters in the Kent distribution
    G = output[0] # \gamma_1 parameters in Kent distribution
    K = output[1].flatten() # \kappa parameter in Kent distribution
    
    loss1 = -1.0 * torch.log(torch.div(K,4*torch.pi*torch.sinh(K)))
    loss2 = -1.0 * ( torch.log(torch.div(K,2*torch.pi)) - K )
    
    mask = K<40.0
    
    mx = masked_tensor(loss1.clone().detach(), mask)
    my = masked_tensor(loss2.clone().detach(), ~mask)
    
    loss_K = torch.where(mask, mx, my)
    
        
    # Compute negative log likelihood using Kent distribution
    loss = torch.mean( loss_K - ( K * torch.sum(G*target,dim=1) ))
    
    
    return loss

# Negative Log Likelihood Loss for HSCDC convnet
# Loss1 is a 15ht order Taylor series about K=0, loss2 is a very close approximation for high K
# To aviod instabilities we either use Loss1 or Loss2, never the true expression for loss
# With this treatement the maximum error on the loss occers when K=2.65, the true loss is 3.5083 and both approximations are off by 0.005
# This method works
def NLLloss_TS(output, target):
    
    # target us the x parameters in the Kent distribution
    G = output[0] # \gamma_1 parameters in Kent distribution
    K = output[1].flatten() # \kappa parameter in Kent distribution
    
    # 15th order taylor series about 0
    loss1 = K**2/6 - K**4/180 + K**6/2835 - K**8/37800 + K**10/467775 - (691* (K**12) )/ 3831077250 + (2 * (K**14))/127702575 + torch.log(torch.tensor(4)*torch.pi)
    # high K approx
    loss2 = -1.0 * ( torch.log(torch.div(K,2*torch.pi)) - K )
    
    loss_K = torch.where(K<2.65, loss1, loss2)
    
        
    # Compute negative log likelihood using Kent distribution
    loss = torch.mean( loss_K  - ( K * torch.sum(G*target,dim=1) ))
    
    
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
                    print("To avoid this, continue training with a lower order Taylor Series in the NLL loss")
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