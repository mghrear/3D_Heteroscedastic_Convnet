# 3D_Heteroscedastic_Convnet
A convolutional neural netwrok for 3D vector direction reconstruction with uncertainty.

## mytools.py
Contain functions which are used in several scripts

## explore_data.ipynb
Exploratory notebook for reading, processing, saving, and loading the data. Contains figures of examples at all stages of the preprocessing.

## process_data.py
This script reads in raw simulation files and process them by doing the following:
1. Applies diffusion to the data
2. Applies a random rotation 
3. Mean-ceneters each simulation

## make_sparse_tensors.py
This script reads the processed data and creates a directory of sparse tensors read for the CNN.
