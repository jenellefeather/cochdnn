import sys
sys.path.append('/om4/group/mcdermott/user/jfeather/projects/robust_audio_networks/robustness')
from robustness.datasets import jsinV3
from robustness.model_utils import make_and_restore_model

# Set direct paths for the system being used
JSIN_PATH = '/om4/group/mcdermott/projects/ibmHearingAid/assets/data/datasets/JSIN_v3.00/nStim_20000/2000ms/rms_0.1/noiseSNR_-10_10/stimSR_20000/reverb_none/noise_all/JSIN_all_v3/subsets'

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
    resume_path = None

    # Resnet Layers Used for Metamer Generation
    metamer_layers = [
         'input_after_preproc',
         'filtered_signal', 
         'spectempfilter_power',
         'avgpool', # Time average power -- what to use for the baseline fMRI predictions
         'final', 
    ]

    # Restore the model
    model, _ = make_and_restore_model(arch='spectemp_filts_time_average_coch1', 
                                      dataset=ds, 
                                      parallel=False,
                                      resume_path=resume_path,
                                      strict=strict,
                                      remap_checkpoint_keys={'model.0.full_rep.rep.Cochleagram.compute_subbands.coch_filters':'coch_filts_old'},
                                     )

    # send the model to the GPU and return it. 
    model.cuda()
    model.eval()

    model.model.filter_freqs = model.model._modules['1'].spectempfilterbank.spec_temp_freqs
    model.model.coarse_define_layers = ['filtered_signal', 'spectempfilter_power']

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
