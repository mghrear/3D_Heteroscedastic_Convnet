#!/bin/bash
#SBATCH --job-name=train_HSCDC
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00 ## time format is DD-HH:MM:SS

#SBATCH --mail-user=majd@hawaii.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80

#SBATCH --output=train_HSCDC_%j.out
#SBATCH --error=train_HSCDC_%j.err

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu

module purge
module load lang/Anaconda3/2022.05 
module load tools/nmap/7.80
source activate mypt
jupyter nbconvert --to=notebook --inplace --ExecutePreprocessor.enabled=True 3D_HSCDC_CNN_tune_kappa.ipynb
