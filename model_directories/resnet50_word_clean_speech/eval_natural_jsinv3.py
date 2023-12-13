import sys
import os
sys.path.append('/om4/group/mcdermott/user/jfeather/projects/robust_audio_networks/robustness/')
sys.path.append('/om4/group/mcdermott/user/jfeather/projects/robust_audio_networks/robustness/plotting_and_analysis_scripts/')
# import make_behavioral_plots

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

ch.manual_seed(0)
np.random.seed(0)

print('Making Loaders Now')
train_loader, val_loader = ds.make_loaders(batch_size=BATCH_SIZE, 
                                           workers=NUM_WORKERS, 
                                           shuffle_train=False, 
                                           shuffle_val=False,
                                           data_aug=True,
                                           subset_type_val='first',
                                           subset_start_val=0,
                                           subset_val=40650,
                                          )

print(len(val_loader))

# Hard-coded base parameters
eval_kwargs = {
    'out_dir': "eval_out",
    'exp_name': "eval_natural_jsinv3",
    'adv_train': 0,
    "adv_eval":0, 
    'constraint': '2',
    'eps': 3,
    'step_size': 1,
    'attack_lr': 1.5,
    'attack_steps': 20,
    'save_ckpt_iters':1,
}

if ds.__dict__.get('multitask_parameters', None) is not None:
    print('CUSTOM LOSSES ARE APPLIED')
    PER_GPU_BATCH_SIZE=int(BATCH_SIZE/ch.cuda.device_count())
    eval_kwargs['custom_train_loss'] = ds.multitask_parameters['custom_loss']
    eval_kwargs['custom_train_loss'].set_batch_size(PER_GPU_BATCH_SIZE)
    from functools import partial
    eval_kwargs['custom_adv_loss'] = partial(ds.multitask_parameters['calc_custom_adv_loss_with_batch_size'], BATCH_SIZE=PER_GPU_BATCH_SIZE)

eval_args = Parameters(eval_kwargs)

# Fill whatever parameters are missing from the defaults
eval_args = defaults.check_and_fill_args(eval_args,
                        defaults.TRAINING_ARGS, ImageNet)
eval_args = defaults.check_and_fill_args(eval_args,
                        defaults.PGD_ARGS, ImageNet)

# Create the cox store, and save the arguments in a table
store = store.Store(eval_args.out_dir, eval_args.exp_name)
print(store)
args_dict = eval_args.as_dict() if isinstance(eval_args, Parameters) else vars(eval_args)
# schema = store.schema_from_dict(args_dict)
store.add_table_like_example('metadata', args_dict)
store['metadata'].append_row(args_dict)

train.eval_model(eval_args, model, val_loader,store=store)
