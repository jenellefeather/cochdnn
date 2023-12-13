from __future__ import division
from scipy.io import wavfile
import os

# make sure we are using the correct plotting display. 
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

import sys
if sys.version_info < (3,):
    from StringIO import StringIO as BytesIO
else:
    from io import BytesIO
import base64

import scipy
import pickle
import h5py
import argparse

import torch
import sys
sys.path.append('../../')
sys.path.append('/om4/group/mcdermott/user/jfeather/projects/robust_audio_networks/robustness/')
from robustness.datasets import jsinV3
from robustness.model_utils import make_and_restore_model
from robustness.audio_functions import audio_transforms

sys.path.append("/home/jfeather/python-packages/tfmatching")
from synthhelpers import load_audio_wav_resample

import itertools

import build_network
import shutil
def preproc_sound_np(sound):
    sound = sound - np.mean(sound)
    sound = sound/np.sqrt(np.mean(sound**2))*0.1
    sound = np.expand_dims(sound, 0)
    sound = torch.from_numpy(sound).float().cuda()
    return sound

############LOAD NETWORK############
model, ds, all_layers = build_network.main(return_metamer_layers=True, strict=False)

##############Begin Define Parameters#################
save_features_dir = './'

if not os.path.isdir(save_features_dir):
    os.mkdir(save_features_dir) 

if not os.path.isdir('test_activations'):
    os.mkdir('test_activations')

#############LOAD_AUDIO################
# contains the metatdata for the list of presented sounds (should be in the correct order)
sound_list = np.load('/om/user/jfeather/neural_predictions/natsounddata/neural_stim_meta.npy')
sound_list = sound_list[0:2]

wavs_location = '/mindhive/mcdermott/shared/Sounds/165_natural_sounds'

SR=20000 # Match with the networks we are building/training
MEASURE_DUR=2
wav_array = np.empty([165, SR*MEASURE_DUR])
for wav_idx, wav_data in enumerate(sound_list):
    shutil.copyfile(os.path.join(wavs_location, wav_data[0].decode('utf-8')), os.path.join('test_activations', '%d_%s'%(wav_idx,wav_data[0].decode('utf-8'))))
    test_audio, SR = load_audio_wav_resample(os.path.join(wavs_location, wav_data[0].decode('utf-8')), DUR_SECS=MEASURE_DUR, resample_SR=SR)
    wav_array[wav_idx,:] = test_audio/np.sqrt(np.mean(test_audio**2))

# Measure the activations for each sound for each layer, and put the input in the dictionary array. 

filename = os.path.join('test_activations', 'nat_sound_test_activations')
all_layers = [e for e in all_layers if 'jittered_relu' not in e] # Don't duplicate these since we aren't synthesizing
net_layer_dict_full = {}
net_h5py_file_full = {} # h5py.File(os.path.join(save_features_dir, filename + '_full.h5'), "w")

# Save the list of layers to the hdf5
net_h5py_file_full['layer_list'] = np.array([layer.encode("utf-8") for layer in all_layers])

for sound_idx, sound_info in enumerate(sound_list):
    orig_wav = wav_array[sound_idx,:]
    sound = preproc_sound_np(wav_array[sound_idx,:])
    with torch.no_grad():
        (predictions, rep, layer_returns), orig_image = model(sound, with_latent=True) # Corresponding representation

    # Make the array have the correct size
    if sound_idx == 0:
        net_layer_dict_full['wav_orig'] = [] # net_h5py_file_full.create_dataset('wav_orig', (len(sound_list), SR*MEASURE_DUR), dtype='float32')
        net_layer_dict_full['wav_preproc'] = []# net_h5py_file_full.create_dataset('wav_preproc', (len(sound_list), SR*MEASURE_DUR), dtype='float32')
        for layer in all_layers:
            print(layer)
            layer_shape_165 = layer_returns[layer].shape
            layer_shape_full = np.array(layer_shape_165) # np.prod(np.array(layer_shape_165))
            print(layer_shape_165)
            net_layer_dict_full[layer] = [] # net_h5py_file_full.create_dataset(layer, (len(sound_list), layer_shape_full), dtype='float32')

    for layer in all_layers:
        net_layer_dict_full[layer].append(layer_returns[layer].cpu().detach().numpy())# .ravel()
        if sound_idx == 0:
            print(layer, '| min:', np.min(net_layer_dict_full[layer][sound_idx]), '| max:',
                  np.max(net_layer_dict_full[layer][sound_idx]))

    net_layer_dict_full['wav_orig'].append(orig_wav)
    net_layer_dict_full['wav_preproc'].append(sound.detach().cpu().numpy())

pickle.dump(net_layer_dict_full, open(os.path.join(save_features_dir, filename + '_full.pckl'), 'wb'))
