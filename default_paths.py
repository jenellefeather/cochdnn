"""
Contains the paths to datasets and model checkpoints directory. Used in other scripts in the repository. 
"""

import os

WORKING_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
MODEL_CHECKPOINT_DIR = os.path.join(WORKING_DIRECTORY, 'model_checkpoints')
MODEL_DIRECTORY = os.path.join(WORKING_DIRECTORY, 'model_directories')
JSIN_PATH = '/om4/group/mcdermott/projects/ibmHearingAid/assets/data/datasets/JSIN_v3.00/nStim_20000/2000ms/rms_0.1/noiseSNR_-10_10/stimSR_20000/reverb_none/noise_all/JSIN_all_v3/subsets'
if not os.path.exists(JSIN_PATH):
    JSIN_PATH = None
    print('### WARNING: UNABLE TO FIND JSIN AUDIO TRAINING DATASET FILES. IF TRAINING AUDIO MODELS, CHANGE PATH SPECIFIED IN analysis_scripts/default_paths.py. MODELS CAN BE LOADED AND TESTED WITHOUT THESE FILES. ###')
