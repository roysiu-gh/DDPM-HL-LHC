#!/bin/bash
#SBATCH --job-name=exc    # create a short name for your job
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=6000
#SBATCH --partition=mpcg.p
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1
##SBATCH --exclusive
#SBATCH --output "./slurm_out/%A.out"
##SBATCH --output "./slurm_out/%A_sample.out"
module purge
module load hpc-env/13.1
module load Python/3.11.3-GCCcore-13.1.0 CUDA/12.4.0
source ~/FYP311/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#nvidia-smi --query-gpu=compute_cap --format=csv

#python3 -m DDPMLHC.model
python3 -m DDPMLHC.sample
