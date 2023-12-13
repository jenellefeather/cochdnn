from robustness.datasets import jsinV3
from robustness.model_utils import make_and_restore_model
from default_paths import *

def make_randomized_state_dict(state_dict, pckl_name='rand_network_indices.pckl',
                               exclude_strings=['preproc.','model.0.full_rep'],
                               rename_dict={'attacker.model.':'model.'}):
    print('#### Randomizing Weights ####')
    import os
    import pickle
    import torch
    import random
    import numpy as np
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    # First check and see if a pickle exists already with the random indices
    if os.path.isfile(pckl_name):
        d_rand_idx = pickle.load(open(pckl_name, 'rb'))
    else:
        d_rand_idx = {}
        for k, v in state_dict.items():
            if any(exclude in k for exclude in exclude_strings):
                continue # Do not shuffle these keys
            if any(exclude in k for exclude in list(rename_dict.keys())):
                continue
            print('Making Random Perm for : %s'%k)
            w = state_dict[k]
            idx = torch.randperm(w.nelement())  # create random indices across all dimensions
            d_rand_idx[k] = idx
        pickle.dump(d_rand_idx, open(pckl_name, 'wb'))

    state_dict_rand = {}
    for k, v in state_dict.items():
       if any(exclude in k for exclude in exclude_strings):
           print(f'________ Using non-permuted weights for {k} ________')
           state_dict_rand[k] = state_dict[k]
       elif any(copy_name in k for copy_name in list(rename_dict.keys())):
           copy_name = [c for c in list(rename_dict.keys()) if c in k]
           assert(len(copy_name)==1)
           copy_name = copy_name[0]
           weight_name_to_copy = k.replace(copy_name, rename_dict[copy_name])
           print(f'________ Loading random indices from permuted architecture layer {weight_name_to_copy} for {k} ________')
           w = state_dict[k]
           idx = d_rand_idx[weight_name_to_copy]
           rand_w = w.view(-1)[idx].view(w.size())  # permute using the stored indices, and reshape back to original shape
           state_dict_rand[k] = rand_w
       else:
           w = state_dict[k]
           # Load random indices
           print(f'________ Loading random indices from permuted architecture for {k} ________')
           idx = d_rand_idx[k]
           rand_w = w.view(-1)[idx].view(w.size())  # permute using the stored indices, and reshape back to original shape
           state_dict_rand[k] = rand_w

    return state_dict_rand

# Make a custom build script for audio_rep_training_cochleagram_1/l2_p1_robust_training
def build_net(include_rep_in_model=True, 
              use_normalization_for_audio_rep=True, 
              ds_kwargs={}, 
              include_identity_sequential=False, 
              return_metamer_layers=False, 
              strict=True):

    # Build the dataset so that the number of classes and normalization 
    # is set appropriately. Not needed for metamer generation, but ds is 
    # used for eval scripts.  
    ds = jsinV3(JSIN_PATH, include_rep_in_model=include_rep_in_model, 
                audio_representation='cochleagram_1',
                use_normalization_for_audio_rep=use_normalization_for_audio_rep, 
                include_identity_sequential=include_identity_sequential, 
                **ds_kwargs) # Sequential will change the state dict names

    # Path to the network checkpoint to load
    resume_path = os.path.join(MODEL_CHECKPOINT_DIR, 'audio_rep_training_cochleagram_1/fmri_paper_more_seeds/standard_training_word_decay_lr_seed2/a64afebf-4bb6-483c-ac0d-29b5af3af486/5_checkpoint.pt')

    # Resnet50 Layers Used for Metamer Generation
    metamer_layers = [
         'input_after_preproc',
         'conv1',
         'bn1',
         'conv1_relu1',
         'maxpool1',
         'layer1',
         'layer2',
         'layer3',
         'layer4',
         'avgpool',
         'final'
    ]

    # Restore the model
    model, _ = make_and_restore_model(arch='resnet50', 
                                      dataset=ds, 
                                      parallel=False,
                                      resume_path=resume_path,
                                      strict=strict
                                     )

    print(model.state_dict()['model.1.bn1.weight'])
    print(model.state_dict()['attacker.model.1.bn1.weight'])
    # Shuffle the weights
    rand_state_dict = make_randomized_state_dict(model.state_dict())
    model.load_state_dict(rand_state_dict)
    print(model.state_dict()['model.1.bn1.weight'])
    print(model.state_dict()['attacker.model.1.bn1.weight'])

    # send the model to the GPU and return it. 
    model.cuda()
    model.eval()

    if return_metamer_layers:
        return model, ds, metamer_layers
    else:
        return model, ds

def main(include_rep_in_model=True,
         use_normalization_for_audio_rep=False,
         return_metamer_layers=False,
         include_identity_sequential=False,
         strict=True,
         ds_kwargs={}):
    # This parameter is not used for this model
#     del include_identity_sequential

    if return_metamer_layers:
        model, ds, metamer_layers = build_net(include_rep_in_model=include_rep_in_model,
                                              use_normalization_for_audio_rep=use_normalization_for_audio_rep,
                                              return_metamer_layers=return_metamer_layers,
                                              strict=strict,
                                              include_identity_sequential=include_identity_sequential,
                                              ds_kwargs=ds_kwargs)
        return model, ds, metamer_layers

    else:
        model, ds = build_net(include_rep_in_model=include_rep_in_model,
                              use_normalization_for_audio_rep=use_normalization_for_audio_rep,
                              return_metamer_layers=return_metamer_layers,
                              strict=strict,
                              include_identity_sequential=include_identity_sequential,
                              ds_kwargs=ds_kwargs)
        return model, ds

if __name__== "__main__":
    main()
