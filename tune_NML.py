import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
import mytools
import mymodels
import csv

########################################
########################################
# Select Energy (35 or 50 keV)
Energy = 50
# Select diff amount (H or L)
Diff = 'H'
#Cheat True/False
cheat = True
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
    
# Copy sparse tensor info dataframe
NML_info = pd.read_pickle('/home/majd/sparse_testing_tensors_'+str(Energy)+'keV_'+Diff+'diff/sparse_tensor_info.pk')
# Promote index to column
NML_info = NML_info.reset_index()

# Add position and charge information for each row
NML_info["positions"] = NML_info.apply(lambda row: torch.load('/home/majd/sparse_testing_tensors_'+str(Energy)+'keV_'+Diff+'diff/'+ 'sparse_recoils_' + str(row['index']) + '.pt').type(torch.FloatTensor).coalesce().indices().int().numpy()*vox_l-eff_l, axis=1)
NML_info["charges"] = NML_info.apply(lambda row: torch.load('/home/majd/sparse_testing_tensors_'+str(Energy)+'keV_'+Diff+'diff/'+ 'sparse_recoils_' + str(row['index']) + '.pt').type(torch.FloatTensor).coalesce().values().flatten().numpy()*1.0, axis=1)
    
    
    
def run(param_list):
        
    # param_list is w_o, Nsigma_low, N_sigma_high
    wo, n_sigma_L, delta_sigma, cheat = param_list
    n_sigma_H = n_sigma_L + delta_sigma
    
    v_pred_temp, v_true_temp, off_true_temp = mytools.test_NML(NML_info, model_NML, n_sigma_L, n_sigma_H, wo, cheat=True)
    
    NML_eff_temp = len(v_pred_temp)/len(NML_info)
    NML_Loss_temp = mytools.CSloss(torch.Tensor(v_pred_temp), torch.Tensor(v_true_temp))
    
    return wo, n_sigma_L, n_sigma_H, NML_Loss_temp, NML_eff_temp
    


    
# Make param list
param_list = []
                
for wo in np.arange(0.01,0.11,0.01):
    for n_sigma_L in np.arange(1.0,3.1,0.1):
        for delta_sigma in np.arange(0.5,3.1,0.1):
                
            param_list.append([wo, n_sigma_L, delta_sigma, cheat])

cpus = mp.cpu_count()
pool = mp.Pool(cpus)

for result in pool.map(run, param_list):
    
    wo, n_sigma_L, n_sigma_H, NML_Loss_temp, NML_eff_temp = result
    
    df_tune = df_tune.append({ 'wo' : wo, 'n_sigma_L' :  n_sigma_L, 'n_sigma_H' : n_sigma_H, 'Loss' : NML_Loss_temp, 'eff' : NML_eff_temp }, ignore_index = True)
    print(result)
    
    
df_tune.to_pickle('tune_NML_'+str(Energy)+'keV_'+Diff+'diff_cheat-'+str(cheat)+'.pk')

