#!/bin/bash
#SBATCH --job-name=met_resnet50
#SBATCH --output=output/standard%A_%a.out
#SBATCH --error=output/standard%A_%a.err
#SBATCH --mem=4000
#SBATCH --time=9:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-399
#SBATCH --constraint="high-capacity&11GB"
#SBATCH --exclude=node093,node040,node094
#SBATCH --partition=mcdermott

module add openmind/miniconda/2020-01-29-py3.7
module add openmind/cudnn/9.1-7.0.5
module add openmind/cuda/9.1

export CONDA_ENVS_PATH=~/my-envs:/om4/group/mcdermott/user/jfeather/conda_envs_files
source activate /om4/group/mcdermott/user/jfeather/conda_envs_files/pytorch
cp ../../../analysis_scripts/make_metamers_wsj400_behavior_only_save_metamer_layers.py .
python make_metamers_wsj400_behavior_only_save_metamer_layers.py $SLURM_ARRAY_TASK_ID -S -I 3000 -N 8
