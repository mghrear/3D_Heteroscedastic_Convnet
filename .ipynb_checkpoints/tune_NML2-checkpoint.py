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
Diff = 'H'

########################################
########################################

# dataframe to store results
df_tune = pd.DataFrame(columns = ["Loss", "epsilon"])

# Load the non-ML model
model_NML = mymodels.NML2

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
for eps in np.arange(0.25,1.0,0.01):
                
    v_pred_temp, v_true_temp, off_true_temp = mytools.test_NML2(NML_info, model_NML, eps)
            
    NML_Loss_temp = mytools.CSloss(torch.Tensor(v_pred_temp), torch.Tensor(v_true_temp)).item()
            
    df_tune = df_tune.append({  'Loss' : NML_Loss_temp, 'epsilon' : eps }, ignore_index = True)
    df_tune.to_pickle('../tune_NML_model/tune_NML2_'+str(Energy)+'keV_'+Diff+'diff.pk')
    
    print('epsilon: ', eps, ' Loss: ', NML_Loss_temp )
            
