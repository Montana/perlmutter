#!/bin/bash
#SBATCH --job-name=mpi_test
#SBATCH --account=your_account_here
#SBATCH --partition=regular
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:10:00
#SBATCH --output=mpi_test.out

module load python
module load cray-mpich
source activate mpi_env

srun python mpi_script.py
