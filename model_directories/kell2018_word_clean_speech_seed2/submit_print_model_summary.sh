#!/bin/bash
#SBATCH --job-name=l2_p3_regression
#SBATCH --output=./output/net_word%A_%a.out
#SBATCH --error=./output/net_word%A_%a.err
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1
#SBATCH --mem=8000
#SBATCH --time=03:30:00
#SBATCH --cpus-per-task=2

module add openmind/cudnn/11.5-v8.3.3.40
module add openmind/cuda/11.3
source activate /om/user/jfeather/.conda/envs/model_metamers_pytorch_update_pytorch

cp /om4/group/mcdermott/user/jfeather/projects/robust_audio_networks/robustness/analysis_scripts_metamers_paper/print_model_summary.py .
python print_model_summary.py
