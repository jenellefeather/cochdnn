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

module add openmind/cudnn/11.5-v8.3.3.40
module add openmind/cuda/11.3
source activate /om/user/jfeather/.conda/envs/model_metamers_pytorch_update_pytorch

cp /om4/group/mcdermott/user/jfeather/projects/robust_audio_networks/robustness/analysis_scripts_metamers_paper/eval_natural_jsinv3.py .

python eval_natural_jsinv3.py
