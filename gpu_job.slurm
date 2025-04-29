#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --account=your_account_here
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:10:00
#SBATCH --output=gpu_test.out

module load python
module load cudatoolkit
source activate conda_env 

srun python gpu_script.py
