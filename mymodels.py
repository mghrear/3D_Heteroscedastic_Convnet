# Script containing all models

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import spconv.pytorch as spconv

    
# Heteroscedastic convnet model
class spConvnet_HSCDC_subM(nn.Module):
    def __init__(self, shape):
        super(spConvnet_HSCDC_subM, self).__init__()
        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels=1, out_channels=32, kernel_size=7, stride=1, bias=True),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels=32, out_channels=40, kernel_size=5, stride=1, bias=True),
            nn.ReLU(),
            spconv.SparseConv3d(in_channels=40, out_channels=50, kernel_size=6, stride=2, bias=True),
            nn.ReLU(),
            spconv.SparseMaxPool3d(kernel_size=2, stride=2),
            spconv.SparseConv3d(in_channels=50, out_channels=30, kernel_size=3, stride=2, bias=True),
            nn.ReLU(),
            spconv.SparseConv3d(in_channels=30, out_channels=10, kernel_size=3, stride=1, bias=True),
            nn.ReLU(),
            spconv.SparseMaxPool3d(kernel_size=2, stride=2),
            spconv.ToDense(),
        )
        self.fc1 = nn.Linear(6**3 *10, 500)
        self.fc2_1 = nn.Linear(500, 200)
        self.fc3_1 = nn.Linear(200, 50)
        self.fc4_1 = nn.Linear(50, 3)
        self.fc2_2 = nn.Linear(500, 200)
        self.fc3_2 = nn.Linear(200, 50)
        self.fc4_2 = nn.Linear(50, 1)
        
        self.shape = shape

    def forward(self, features, indices, batch_size):
        
        x_sp = spconv.SparseConvTensor(features, indices, self.shape, batch_size)
        
        x = self.net(x_sp)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x1 = F.relu(self.fc2_1(x))
        x1 = torch.tanh(self.fc3_1(x1))
        output1 = F.normalize(self.fc4_1(x1),dim=1)
        x2 = F.relu(self.fc2_2(x))
        x2 = F.relu(self.fc3_2(x2))
        output2 = F.softplus(self.fc4_2(x2))
                
        return output1,output2

    
# Homoscedastic convnet model    
class spConvnet_subM (nn.Module):
    def __init__(self, shape):
        super(spConvnet_subM, self).__init__()
        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels=1, out_channels=32, kernel_size=7, stride=1, bias=True),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels=32, out_channels=40, kernel_size=5, stride=1, bias=True),
            nn.ReLU(),
            spconv.SparseConv3d(in_channels=40, out_channels=50, kernel_size=6, stride=2, bias=True),
            nn.ReLU(),
            spconv.SparseMaxPool3d(kernel_size=2, stride=2),
            spconv.SparseConv3d(in_channels=50, out_channels=30, kernel_size=3, stride=2, bias=True),
            nn.ReLU(),
            spconv.SparseConv3d(in_channels=30, out_channels=10, kernel_size=3, stride=1, bias=True),
            nn.ReLU(),
            spconv.SparseMaxPool3d(kernel_size=2, stride=2),
            spconv.ToDense(),
        )
        self.fc1 = nn.Linear(6**3 *10, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 50)
        self.fc4 = nn.Linear(50, 3)
        
        self.shape = shape

    def forward(self, features, indices, batch_size):
        
        x_sp = spconv.SparseConvTensor(features, indices, self.shape, batch_size)
                        
        x = self.net(x_sp)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        output = F.normalize(self.fc4(x),dim=1)
        
        return output
    
    
# Method for determining the initial direction of an electron recoil without ML
# This method is from https://iopscience.iop.org/article/10.3847/1538-3881/ac51c9/pdf
# Useful ref for weighted SVD fits: https://lucidar.me/en/mathematics/weighted-pca/
def NML(x_vals, y_vals, z_vals, charges, true_dir, n_sigma_L = 1.5, n_sigma_H = 3, w_o = 0.05, cheat = False):

    X = np.array([x_vals,y_vals,z_vals]).T

    # 1) Center on barycenter
    # Barycenter is the charge-weighted mean position
    x_b = np.sum(X*(charges.reshape(len(charges),1)),axis=0)/np.sum(charges)
    # Shift data to barycenter
    X = X-x_b

    # 2) Find principle axis
    # Use charges for weights
    W = charges.reshape(len(charges),1)
    # Compute weighted covariance matrix
    WCM = ( (W*X).T @ X ) / np.sum(W)
    U1,S1,D1 =  np.linalg.svd(WCM)
    v_PA = np.array([D1[0][0],D1[0][1],D1[0][2]])
    
    # 2)a. Compute second moment about principle axis 
    M_2 = S1[0]
    # 2)b. Compute third moment about principle axis
    X_proj = X@v_PA
    M_3 = np.sum( charges * ( X_proj**3 ) ) / np.sum(charges)

    # 3) keep only points where projected sign is same as sgn(M3) and n_sigma_L M_2 <|proj_i| < n_sigma_H M2
    X_IR = X [ (np.sign(X_proj) == np.sign(M_3)) & ( (n_sigma_L*M_2) <  np.abs(X_proj) ) & (np.abs(X_proj) < (n_sigma_H*M_2))]
    charges_IR = charges [ (np.sign(X_proj) == np.sign(M_3)) & ( (n_sigma_L*M_2) <  np.abs(X_proj) ) & (np.abs(X_proj) < (n_sigma_H*M_2))]
    
    # If there are not enough points in the predicted interaction region, this method fails
    # We return a False flag for this case
    if len(charges_IR) < 2:
        return np.array([0,0,0]), False
    
    else:
        # 4) find the interaction point
        x_IP = np.sum(X_IR*(charges_IR.reshape(len(charges_IR),1)),axis=0)/np.sum(charges_IR)

        # 5) Find the final direction
        # Center on interaction point
        X_IR = X_IR-x_IP
        # re-weight charges
        charges_IR = charges_IR * np.exp(-1*np.linalg.norm(X_IR,axis=1)/w_o)
        # Re-shape weights
        W = charges_IR.reshape(len(charges_IR),1)
        # computed weighted covariance matrix
        WCM = ( (W*X_IR).T @ X_IR ) / np.sum(W)
        # run SVD
        U2,S2,D2 = np.linalg.svd(WCM)

        v_IP = np.array([D2[0][0],D2[0][1],D2[0][2]])

        # Assign v_PA the correct head/tail (based on skewness along principal axis)
        v_PA = -1.0*np.sign(M_3)*v_PA
        
        # Assign head-tail on final direction, use true direction to make assignment if cheat = true
        if cheat == True:
            v_IP = np.sign( np.dot(v_IP,true_dir) ) * v_IP
        else:
            v_IP = np.sign( np.dot(v_IP,v_PA) ) * v_IP
            

        # Return initial direction prediction and True flag
        return v_IP, True
    

# Method for determining the initial direction of an electron recoil without ML
# This is a "cheat" method which uses information that is not experimentally available
def NML2 (x_vals, y_vals, z_vals, charges, true_dir, offset, eps):
    
    # Shift starting point of recoil to origin
    x_vals = x_vals - offset[0]
    y_vals = y_vals - offset[1]
    z_vals = z_vals - offset[2]
    
    # Idenitfy points within eps of origin
    T_arr = np.sqrt(x_vals**2+y_vals**2+z_vals**2) < eps
    
    # Keep only points within eps of origin
    charges = charges[T_arr]
    X = np.array([x_vals[T_arr],y_vals[T_arr],z_vals[T_arr]]).T
    
    # Center on barycenter
    x_b = np.sum(X*(charges.reshape(len(charges),1)),axis=0)/np.sum(charges)
    
    # Shift data to barycenter
    X = X-x_b
    
    # Find principle axis
    # Use charges for weights
    W = charges.reshape(len(charges),1)
    # Compute weighted covariance matrix
    WCM = ( (W*X).T @ X ) / np.sum(W)
    U1,S1,D1 =  np.linalg.svd(WCM)
    v_PA = np.array([D1[0][0],D1[0][1],D1[0][2]])
    
    # Assign correct head-tail to the principle axis
    v_PA = np.sign( np.dot(v_PA,true_dir) ) * v_PA

    return v_PA
