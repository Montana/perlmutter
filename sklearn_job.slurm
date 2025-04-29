#!/bin/bash
#SBATCH --job-name=rf_test
#SBATCH --account=your_account_here
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=rf_test.out

module load python
source activate your_env_name_here

srun python rf_test.py
