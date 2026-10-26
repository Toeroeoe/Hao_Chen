#!/bin/bash -x
#SBATCH --job-name=pft_co2_all
#SBATCH --account=jicg41
#SBATCH --ntasks=11
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

source activate mlspinup
cd /p/scratch/cjicg41/chenh/ML_spinup


srun -n 1 python tensorflow_nn_pft_final_pco2_240_7.py &
srun -n 1 python tensorflow_nn_pft_final_pco2_240_15.py &
srun -n 1 python tensorflow_nn_pft_final_pco2_240_16.py &
wait