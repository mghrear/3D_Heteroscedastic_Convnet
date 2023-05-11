import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import spconv.pytorch as spconv


class spConvnet_HSCDC(nn.Module):
    def __init__(self, shape):
        super(spConvnet_HSCDC, self).__init__()
        self.net = spconv.SparseSequential(
            spconv.SparseConv3d(in_channels=1, out_channels=50, kernel_size=6, stride=2, bias=True),
            nn.ReLU(),
            spconv.SparseMaxPool3d(kernel_size=2, stride=2),
            spconv.SparseConv3d(in_channels=50, out_channels=30, kernel_size=4, stride=1, bias=True),
            nn.ReLU(),
            spconv.SparseConv3d(in_channels=30, out_channels=20, kernel_size=3, stride=1, bias=True),
            nn.ReLU(),
            spconv.SparseMaxPool3d(kernel_size=2, stride=2),
            spconv.ToDense(),
        )
        self.fc1 = nn.Linear(12**3 *20, 100)
        self.fc2_1 = nn.Linear(100, 30)
        self.fc3_1 = nn.Linear(30, 3)
        self.fc2_2 = nn.Linear(100, 30)
        self.fc3_2 = nn.Linear(30, 1)
        
        self.shape = shape

    def forward(self, features, indices, batch_size):
        
        x_sp = spconv.SparseConvTensor(features, indices, self.shape, batch_size)
        
        x = self.net(x_sp)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x1 = torch.tanh(self.fc2_1(x))
        output1 = F.normalize(self.fc3_1(x1),dim=1)
        x2 = F.relu(self.fc2_2(x))
        # Training network to predict log(K) is more numerically stable
        output2 = torch.log(F.softplus(self.fc3_2(x2)))
                
        return output1,output2
    
    
class spConvnet(nn.Module):
    def __init__(self, shape):
        super(spConvnet, self).__init__()
        self.net = spconv.SparseSequential(
            spconv.SparseConv3d(in_channels=1, out_channels=50, kernel_size=6, stride=2, bias=True),
            nn.ReLU(),
            spconv.SparseMaxPool3d(kernel_size=2, stride=2),
            spconv.SparseConv3d(in_channels=50, out_channels=30, kernel_size=4, stride=1, bias=True),
            nn.ReLU(),
            spconv.SparseConv3d(in_channels=30, out_channels=20, kernel_size=3, stride=1, bias=True),
            nn.ReLU(),
            spconv.SparseMaxPool3d(kernel_size=2, stride=2),
            spconv.ToDense(),
        )
        self.fc1 = nn.Linear(12**3 *20, 100)
        self.fc2 = nn.Linear(100, 30)
        self.fc3 = nn.Linear(30, 3)
        
        self.shape = shape

    def forward(self, features, indices, batch_size):
        
        x_sp = spconv.SparseConvTensor(features, indices, self.shape, batch_size)
                        
        x = self.net(x_sp)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        output = F.normalize(self.fc3(x),dim=1)
        
        return output
    
    
# Method for determining the initial direction of an electron recoil without ML
# refs: https://iopscience.iop.org/article/10.3847/1538-3881/ac51c9/pdf , https://lucidar.me/en/mathematics/weighted-pca/
def NML(x_vals, y_vals, z_vals, charges, true_dir, n_sigma_L = 1.5, n_sigma_H = 3, w_o = 0.05, cheat = False):

    X = np.array([x_vals,y_vals,z_vals]).T

    # 1) Center on barycenter
    # Barycenter is the charge-weighted mean position
    x_b = np.sum(X*(charges.reshape(len(charges),1)),axis=0)/np.sum(charges)
    # Shift data to barycenter
    X = X-x_b

    # 2) Find principle axis
    U1,S1,D1 =  np.linalg.svd(X)
    v_PA = np.array([D1[0][0],D1[0][1],D1[0][2]])
    
    # 2)a. Compute second moment about principle axis 
    X_proj = X@v_PA
    M_2 = np.sum( charges * ( X_proj**2 ) ) / np.sum(charges)
    # 2)b. Compute third moment about principle axis 
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