import numpy as np
import pandas as pd
import torch
import mytools
import mymodels

########################################
########################################
# Select Energy (40 or 50 keV)
Energy = 40
# Select diff amount (H or L)
Diff = 'L'
# Slecet cheat parameter True or False
cheat = False
########################################
########################################

# dataframe to store results
df_tune = pd.DataFrame(columns = ["wo", "n_sigma_L", "n_sigma_H", "Loss" , "eff"])

# Load the non-ML model
model_NML = mymodels.NML

# x/y/z length being kept in cm
eff_l= mytools.voxel_grid['eff_l']
# Voxel size in cm
vox_l = mytools.voxel_grid['vox_l']
    
# Read sparse tensor info dataframe
NML_info = pd.read_pickle('/home/majd/sparse_testing_tensors_'+str(Energy)+'keV_'+Diff+'diff/sparse_tensor_info.pk')
# Promote index to column
NML_info = NML_info.reset_index()

# Add physical position and charge information for each row
NML_info["positions"] = NML_info.apply(lambda row: torch.load('/home/majd/sparse_testing_tensors_'+str(Energy)+'keV_'+Diff+'diff/'+ 'sparse_recoils_' + str(row['index']) + '.pt').type(torch.FloatTensor).coalesce().indices().int().numpy()*vox_l-eff_l, axis=1)
NML_info["charges"] = NML_info.apply(lambda row: torch.load('/home/majd/sparse_testing_tensors_'+str(Energy)+'keV_'+Diff+'diff/'+ 'sparse_recoils_' + str(row['index']) + '.pt').type(torch.FloatTensor).coalesce().values().flatten().numpy()*1.0, axis=1)

# Do a grid search for the best parameters
# for wo in np.arange(0.04,0.50,0.02):
#     for n_sigma_L in np.arange(1.4,5.0,0.2):
#         for delta_sigma in np.arange(1.4,5.0,0.2):

# Coarse search settings
wos = np.arange(0.02,2.0,0.2)
wos = np.append(wos,[1e999]) # add wo = inf to the end of the array, this tests the case where we do not re-weight the points with exp(-dist(x_i,x_ip)/wo)
n_sigma_Ls = np.arange(1.0,8.0,0.5)
delta_sigmas = np.arange(1.0,15.0,0.5)

cnt = 0

for wo in wos:
    for n_sigma_L in n_sigma_Ls:
        for delta_sigma in delta_sigmas:
            
            n_sigma_H = n_sigma_L + delta_sigma
            
            v_pred_temp, v_true_temp, off_true_temp = mytools.test_NML(NML_info, model_NML, n_sigma_L, n_sigma_H, wo, cheat)
            
            NML_eff_temp = len(v_pred_temp)/len(NML_info)
            NML_Loss_temp = mytools.CSloss(torch.Tensor(v_pred_temp), torch.Tensor(v_true_temp)).item()
            
            df_tune = df_tune.append({ 'wo' : wo, 'n_sigma_L' :  n_sigma_L, 'n_sigma_H' : n_sigma_H, 'Loss' : NML_Loss_temp, 'eff' : NML_eff_temp }, ignore_index = True)
            df_tune.to_pickle('../tune_NML_model_coarse/tune_NML_'+str(Energy)+'keV_'+Diff+'diff_cheat-'+str(cheat)+'.pk')
            print(cnt, "/", len(wos)*len(n_sigma_Ls)*len(delta_sigmas) )
            print('wo: ', wo, ' n_sigma_L: ',  n_sigma_L, ' n_sigma_H: ', n_sigma_H, ' Loss: ', NML_Loss_temp, ' eff: ', NML_eff_temp )
            cnt += 1
            
