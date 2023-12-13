import sys
import os
sys.path.append('/om4/group/mcdermott/user/jfeather/projects/robust_audio_networks/robustness/')
sys.path.append('/om4/group/mcdermott/user/jfeather/projects/robust_audio_networks/robustness/plotting_and_analysis_scripts/')
import make_behavioral_plots
from torchsummary import summary

import build_network

from robustness.datasets import ImageNet

from robustness import train
from cox.utils import Parameters
from cox import store

from robustness import model_utils, datasets, train, defaults
import torch as ch 
import numpy as np

BATCH_SIZE=16
NUM_WORKERS=4

model, ds = build_network.main()

print('Making Loaders Now')
train_loader, val_loader = ds.make_loaders(batch_size=BATCH_SIZE,
                                           workers=NUM_WORKERS,
                                           shuffle_train=False,
                                           shuffle_val=False,
                                           data_aug=True,
                                           subset_type_val='first',
                                           subset_start_val=0,
                                           subset_val=1,
                                           subset=1,
                                           subset_start=0,
                                          )

batch_1 = next(iter(train_loader))

summary(model.model, batch_1[0][0].shape)
