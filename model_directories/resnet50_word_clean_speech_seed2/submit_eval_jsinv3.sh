#!/bin/bash
#SBATCH --job-name=behavior_plot
#SBATCH --output=output/behavior_plot%j.out
#SBATCH --error=output/behavior_plot%j.err
#SBATCH --mem=16000
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=11GB
#SBATCH --partition=mcdermott

module add openmind/miniconda/2020-01-29-py3.7
module add openmind/cudnn/9.1-7.0.5
module add openmind/cuda/9.1

# cp /om4/group/mcdermott/user/jfeather/projects/robust_audio_networks/robustness/analysis_scripts_metamers_paper/eval_natural_jsinv3.py .

export CONDA_ENVS_PATH=~/my-envs:/om4/group/mcdermott/user/jfeather/conda_envs_files
source activate /om4/group/mcdermott/user/jfeather/conda_envs_files/pytorch
python eval_natural_jsinv3.py
