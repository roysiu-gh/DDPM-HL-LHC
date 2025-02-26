#!/bin/bash
#SBATCH --job-name=DDPM    # create a short name for your job
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=3700
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH -o "./slurm_out/%A.out"
module purge
module load GCCcore/11.2.0 Python/3.9.6
# module load GCC/12.3.0 OpenMPI/4.1.5 Python/3.11.3 CUDA/12.4.0
# source ~/FYP311/bin/activate
source DDPM-HL-LHC/venv/bin/activate
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#nvidia-smi --query-gpu=compute_cap --format=csv

python3 -m DDPMLHC.main_rs