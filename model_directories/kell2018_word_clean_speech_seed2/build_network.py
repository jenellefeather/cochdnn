from robustness.datasets import jsinV3
from robustness.model_utils import make_and_restore_model
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
                **ds_kwargs) # Sequential will change the state dict names

    # Path to the network checkpoint to load
    resume_path = os.path.join(MODEL_CHECKPOINT_DIR, 'audio_rep_training_cochleagram_1/clean_speech_training/kell2018_decay_lr_clean_speech_seed2/2e829114-0fbc-476a-94fa-a2aaa686605a/5_checkpoint.pt' )

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
     'final',
    ]

    # Restore the model
    model, _ = make_and_restore_model(arch='kell2018', 
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
