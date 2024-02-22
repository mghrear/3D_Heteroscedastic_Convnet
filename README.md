# 3D_Heteroscedastic_Convnet
A CNN for probabilistic 3D direction predicitions.

## mytools.py
Contains functions, classes, and dictionaries that are used throughout.

## mymodels.py
Contains all models, which includes:
1. spConvnet_HSCDC_subM: A heteroscedastic convnet for probabilistic direction predictions
2. spConvnet_subM: A homoscedastic convnet for point estimate predictions
3. NML: A non-ML algorithim originally developed for X-ray polarimetry
4. NML2: A "cheat" method which uses information that is only available in simulation to estimate the best expected performance

## explore_data.ipynb
An exploratory notebook for reading, processing, saving, and loading the electron simulation data. Contains figures / examples throughout all stages of the preprocessing.

## process_data.py
This script reads in raw simulations (stored as root files) and process them (into pickle files), doing the following:
1. Applies diffusion to the data
2. Applies a random rotation 
3. Mean-ceneters each simulation

## make_sparse_tensors.py
This script reads the pickle files from process_data.py data and outputs PyTorch sparse tensors in the COO format. It also outputs pickle files that store the coresponding labels (true directions), and other information (offset, energy, applied diffusion).

## arrow_generator.ipynb
Defines and generates a data set of arrows pointing in random direction for the toy test case to demonstrate the model is general. 

## explain_nan_issue.ipynb
A jupyter notebook that illustrates some of the practicle issues encountered while training the heterscedastic model

## RCN_arrows.ipynb
Notebook where we train a regular (homoscedastic) convnet on a toy test case (arrows). Here the convnet quickly learns to predict the directions of 3D arrows.

## HCN_arrows.ipynb
Notebook where we train a heteroscedastic convnet on a toy test case (arrows). Here the convnet quickly learns to predict the direction of 3D arrows with a very small directional uncertainty.

## RCN.ipynb
Notebook where we train the regular (homoscedastic) convnet on electron simulations.

## HCN.ipynb
Notebook where we train the heteroscedastic convnet on electron simulations.

## tune_NML.py
A script used to tune the parameters of the non-ML algorithim originally developed for X-ray polarimetry.

## tune_NML2.py
A script used to tune te "cheat" method which uses information that is only available in simulation to estimate the best expected performance

## test_models.ipynb
A notebook for testing and comparing all models on the electron simulation test sets.

## test_arrows.ipynb
A notebook for testing the RCN and HCN models on the toy test case of detecting arrow directions.

## test_kappa.ipynb
A notebook for testing the calibration of the HCN model.

## make_NNplots.py
A python script that creates figures depicting the dense portions of the RCN and HCN models.

## plot_loss_functions.ipynb
A notebook that plots loss function to illustrate the issue with predicting distribtion over the theta and phi angles seperately. Our framework avoids this by predicting distributions on S2.

## submit_job.slurm
A slurm script for queuing jobs that run a jupyter notebook.
