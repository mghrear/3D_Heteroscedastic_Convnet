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

## 3D_CNN_arrows.ipynb
Notebook where I train the homoscedastic convnet on the simple arrow test case.

## 3D_HSCDC_arrows.ipynb
Notebook where I train the heteroscedastic convnet on the simple arrow test case.

## 3D_CNN.ipynb
Notebook where I train the homoscedastic convnet on electron recoils.

## 3D_HSCDC.ipynb
Notebook where I train the heteroscedastic convnet on electron recoils.

## tune_NML.py
A script used to tune the parameters of the non-ML model via gridsearch.

## test_models.ipynb
A notebook for testing and comparing all models.

## test_arrows.ipynb


## test_kappa.ipynb