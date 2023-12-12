#!/bin/bash
#SBATCH --job-name=behavior_plot
#SBATCH --output=output/behavior_plot%j.out
#SBATCH --error=output/behavior_plot%j.err
#SBATCH --mem=50000
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --constraint=high-capacity
#SBATCH --exclude=node093,node094,node040,node097,node098
#SBATCH --partition=mcdermott


module add openmind/cudnn/11.5-v8.3.3.40
module add openmind/cuda/11.3
source activate /om/user/jfeather/.conda/envs/model_metamers_pytorch_update_pytorch

cp ../../../plotting_and_analysis_scripts/make_esc_model_plots.py .

python make_esc_model_plots.py -D /scratch/scratch/Fri/jfeather -L -3 -A 4096 -R 5 -P -O -C 0.01 0.1 1 10 100
