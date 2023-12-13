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
                include_all_labels=True,
                **ds_kwargs) # Sequential will change the state dict names

    # Path to the network checkpoint to load
    resume_path = os.path.join(MODEL_CHECKPOINT_DIR, 'audio_rep_training_cochleagram_1/clean_speech_training/standard_training_word_decay_lr_clean_speech_speaker_seed2/train_out/5_checkpoint.pt')

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
         'final/signal/speaker_int',
    ]

    TASK_LOSS_PARAMS={}
    # Speaker task is included for this network
    TASK_LOSS_PARAMS['signal/speaker_int'] = {
        'loss_type': 'crossentropyloss',
        'weight': 1.0
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
    model, _ = make_and_restore_model(arch='resnet_multi_task50', 
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
