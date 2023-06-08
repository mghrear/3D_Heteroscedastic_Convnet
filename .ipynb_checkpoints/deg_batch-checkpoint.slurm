#!/bin/bash                                                                                                                      
#SBATCH --job-name=train_HSCDC 
#SBATCH --partition=gpu                                                                                                       
## 3 day max run time for public partitions, except 4 hour max runtime for the sandbox partition                                  
#SBATCH --time=3-00:00:00 ## time format is DD-HH:MM:SS 
#SBATCH --cpus-per-task=2                                                                           
#SBATCH --mem=20G ## max amount of memory per node you require
#SBATCH --gpus=1 ## count of GPUs required for the jobe
##SBATCH --core-spec=0 ## Uncomment to allow jobs to request all cores on a node
#SBATCH --error=hello-%A.err ## %A - filled with jobid                                                                            
#SBATCH --output=hello-%A.out ## %A - filled with jobid                                                                           
## Useful for remote notification                                                                                                 
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=majd@hawaii.edu                              
## All options and environment variables found on schedMD site: http://slurm.schedmd.com/sbatch.html

module purge
module load lang/Anaconda3/2022.05 
module load tools/nmap/7.80
source activate mypt
jupyter nbconvert --to=notebook --inplace --ExecutePreprocessor.enabled=True 3D_HSCDC_CNN_subM-256.ipynb

