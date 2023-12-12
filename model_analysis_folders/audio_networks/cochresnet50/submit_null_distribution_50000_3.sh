#!/bin/bash
#SBATCH --job-name=null_vgg
#SBATCH --output=output/null_%A_%a.out
#SBATCH --error=output/null_%A_%a.err
#SBATCH --mem=128000
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1
#SBATCH --array=0
#SBATCH --constraint=high-capacity
#SBATCH --partition=mcdermott

module add openmind/miniconda/2020-01-29-py3.7
module add openmind/cudnn/9.1-7.0.5
module add openmind/cuda/9.1

cp ../../../analysis_scripts/make_null_distributions.py .

export CONDA_ENVS_PATH=~/my-envs:/om4/group/mcdermott/user/jfeather/conda_envs_files
source activate /om4/group/mcdermott/user/jfeather/conda_envs_files/pytorch
# Run 5 different starting index to grab from different locations in the wsn dataset.
# Training data has more than 2000000 samples so we will never be repeating pairs.
python make_null_distributions.py -N 50000 -I 10 -R 0 --no-shuffle &
python make_null_distributions.py -N 50000 -I 11 -R 0 --no-shuffle &
python make_null_distributions.py -N 50000 -I 12 -R 0 --no-shuffle &
python make_null_distributions.py -N 50000 -I 13 -R 0 --no-shuffle &
python make_null_distributions.py -N 50000 -I 14 -R 0 --no-shuffle &

wait
