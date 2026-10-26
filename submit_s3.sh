#!/bin/bash -x
#SBATCH --job-name=soil3
#SBATCH --account=jicg41 
#SBATCH --ntasks=1
#SBATCH --time=24:00:00

source activate mlspinup
cd /p/scratch/cjicg41/chenh/ML_spinup

srun -n 1 --mem 1024000 python tensorflow_nn_soil_final_3.py
