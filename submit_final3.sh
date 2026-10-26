#!/bin/bash -x
#SBATCH --job-name=pft_co2_all
#SBATCH --account=jicg41
#SBATCH --ntasks=11
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

source activate mlspinup
cd /p/scratch/cjicg41/chenh/ML_spinup


srun -n 1 python tensorflow_nn_pft_final_pco2_240_10.py &
srun -n 1 python tensorflow_nn_pft_final_pco2_240_8.py &
srun -n 1 python tensorflow_nn_pft_final_pco2_240_13.py &
srun -n 1 python tensorflow_nn_pft_final_pco2_240_14.py &
wait