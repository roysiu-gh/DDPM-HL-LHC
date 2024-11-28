#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2679
#SBATCH --time=01:00:00
#SBATCH --output=/storage/physics/phuftc/ddpm_runs/%A.out
#SBATCH --job-name=ddpm

cd $SLURM_SUBMIT_DIR

module purge
module load GCCcore/11.3.0 Python/3.10.4 texlive

source ~/venv2/bin/activate
python3 ./src/resolution_plots.py