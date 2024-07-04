# 3D_Heteroscedastic_Convnet
A CNN for probabilistic 3D direction predicitions.

## mytools.py
Contains functions, classes, and dictionaries that are used throughout.

## mymodels.py
Contains all models, which includes:
1. spConvnet_HSCDC_subM: A heteroscedastic convnet for probabilistic direction predictions (creates vMF-NN or Gauss-NN model, depending on loss function used)
2. spConvnet_subM: A homoscedastic convnet for point estimate predictions (creates Det-NN model)
3. NML: A non-ML algorithim originally developed for X-ray polarimetry (Non-ML model)
4. NML2: A method which uses information that is only available in simulation to estimate the best expected performance (Best-Expected model)

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

## Det-NN_arrows.ipynb
Notebook where we train the Det-NN model on a toy test case (arrows). Here the convnet quickly learns to predict the directions of 3D arrows.

## vMF-NN_arrows.ipynb
Notebook where we train the vMF-NN on a toy test case (arrows). Here the convnet quickly learns to predict the direction of 3D arrows with a very small directional uncertainty.

## Det-NN.ipynb
Notebook where we train the Det-NN model on electron simulations.

## vMF-NN.ipynb
Notebook where we train the vMF-NN model on electron simulations.

## Gauss-NN.ipynb
Notebook where we train the Gauss-NN model on electron simulations.

## tune_NML.py
A script used to tune the parameters of the non-ML algorithim originally developed for X-ray polarimetry.

## tune_Best-Expected.py
A script used to tune the method which uses information that is only available in simulation to estimate the best expected performance.

## test_models.ipynb
A notebook for testing and comparing models on the electron simulation test sets.

## test_arrows.ipynb
A notebook for testing Det-NN and vMF-NN on the toy problem of detecting arrow directions.

## test_kappa.ipynb
A notebook for testing the calibration of the vMF-NN model.

## make_NNplots.py
A python script that creates figures depicting the dense portions of the Det-NN and vMF-NN models.

## submit_job.slurm
A slurm script for queuing jobs that run a jupyter notebook.

# Citation
If you find this useful to your work, please cite

```latex
@article{10.1088/2632-2153/ad5f13,
	author={Ghrear, Majd and Sadowski, Peter and Vahsen, Sven Einar},
	title={Deep Probabilistic Direction Prediction in 3D with Applications to Directional Dark Matter Detectors},
	journal={Machine Learning: Science and Technology},
	url={http://iopscience.iop.org/article/10.1088/2632-2153/ad5f13},
	year={2024}
}
```
