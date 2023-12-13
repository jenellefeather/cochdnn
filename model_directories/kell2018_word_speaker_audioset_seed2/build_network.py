from robustness.datasets import jsinV3
from robustness.model_utils import make_and_restore_model
from robustness.audio_functions import jsinV3_loss_functions
from default_paths import *

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
                eval_max=8,
                include_all_labels=True,
                **ds_kwargs) # Sequential will change the state dict names

    # Path to the network checkpoint to load
    resume_path = os.path.join(MODEL_CHECKPOINT_DIR, 'audio_rep_training_cochleagram_1/fmri_paper_more_seeds/kell2018_word_audioset_speaker_decay_lr_seed2/ef1d3dc2-c133-4206-9cf5-9aa82706235b/5_checkpoint.pt')

    # Resnet Layers Used for Metamer Generation
    metamer_layers = [
     'input_after_preproc',
     'batchnorm0',
     'conv0',
     'relu0',
     'maxpool0',
     'batchnorm1',
     'conv1',
     'relu1',
     'maxpool1',
     'batchnorm2',
     'conv2',
     'relu2',
     'conv3',
     'relu3',
     'conv4',
     'relu4',
     'avgpool',
     'fullyconnected',
     'relufc',
     'dropout',
     'final/signal/word_int',
     'final/signal/speaker_int',
     'final/noise/labels_binary_via_int',
    ]

    TASK_LOSS_PARAMS={}
    TASK_LOSS_PARAMS['signal/word_int'] = {
        'loss_type': 'crossentropyloss',
        'weight': 1.0
    }
    TASK_LOSS_PARAMS['noise/labels_binary_via_int'] = {
        'loss_type': 'bcewithlogitsloss',
        'weight': 300.0
    }
    TASK_LOSS_PARAMS['signal/speaker_int'] = {
        'loss_type': 'crossentropyloss',
        'weight': 0.25
    }

    PLACEHOLDER_BATCH_SIZE=None
    custom_loss = jsinV3_loss_functions.jsinV3_multi_task_loss(TASK_LOSS_PARAMS, PLACEHOLDER_BATCH_SIZE).cuda()
    custom_adv_criterion = jsinV3_loss_functions.jsinV3_multi_task_loss(TASK_LOSS_PARAMS, PLACEHOLDER_BATCH_SIZE, reduction='none')

    def calc_custom_adv_loss_with_batch_size(model, inp, target, BATCH_SIZE):
        '''
        Wraps the adversarial criterion to take in the model.
        '''
        output = model(inp)
        custom_adv_criterion.set_batch_size(BATCH_SIZE)
        loss = custom_adv_criterion(output, target)
        return loss, output

    # Store the custom loss parameters within ds, so that we return it
    ds.multitask_parameters = {'TASK_LOSS_PARAMS': TASK_LOSS_PARAMS,
                               'custom_loss': custom_loss,
                               'custom_adv_criterion': custom_adv_criterion,
                               'calc_custom_adv_loss_with_batch_size': calc_custom_adv_loss_with_batch_size}

    # Restore the model
    model, _ = make_and_restore_model(arch='kell2018_multi_task', 
                                      dataset=ds, 
                                      parallel=False,
                                      resume_path=resume_path,
                                      strict=strict
                                     )

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
