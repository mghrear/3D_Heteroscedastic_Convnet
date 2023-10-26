# 3D_Heteroscedastic_Convnet
A convolutional neural netwrok for 3D vector direction reconstruction with uncertainty.

## mytools.py
Contains functions, classes, and dictionaries that are used throughout.

## mymodels.py
Contains all models, which includes:
1. A heteroscedastic convnet
2. A homoscedastic convnet
3. A non-ML algorithim originally developed for X-ray polarimetry
4. A "cheat" method which uses information that is not experimentally available

## explore_data.ipynb
An exploratory notebook for reading, processing, saving, and loading the data. Contains figures / examples throughout all stages of the preprocessing.

## process_data.py
This script reads in raw simulations (stored in root files) and process them (into pickle files), doing the following:
1. Applies diffusion to the data
2. Applies a random rotation 
3. Mean-ceneters each simulation

## make_sparse_tensors.py
This script reads the pickle files from process_data.py data and outputs PyTorch sparse tensors in the COO format. It also outputs pickle files that store the coresponding labels (true directions), and other information (offset, energy, applied diffusion).

## arrow_generator.ipynb
Generates arrows for a simple test case

## explain_nan_issue.ipynb
A jupyter notebook that illustrates some of the practicle issues encountered while training our models

## RCN_arrows.ipynb
Notebook where we train a regular (homoscedastic) convnet on a simple test case. Here the convnet quickly learns to predict the directions of 3D arrows.

## HCN_arrows.ipynb
Notebook where we train a heteroscedastic convnet on a simple test case. Here the convnet quickly learns to predict the direction of 3D arrows with a very small directional uncertainty.

## RCN.ipynb
Notebook where we train the regular (homoscedastic) convnet on electron recoils.

## HCN.ipynb
Notebook where we train the heteroscedastic convnet on electron recoils.

## tune_NML.py
A script used to tune the parameters of the non-ML model via gridsearch.

## tune_NML2.py
A script used to tune the epsilon parameter of the "cheat" method.

## test_models.ipynb
A notebook for testing and comparing all models.

## test_arrows.ipynb
A notebook for testing the RCN and HCN models on the simple case of detecting arrow directions.

## test_kappa.ipynb
A notebook for testing the uncertainty predictions of the HCN model.

## make_NNplots.py
A python script for illustrating the dense portions of the RCN and HCN models.

## submit_job.slurm
A slurm script for queuing jobs that run a jupyter notebook.