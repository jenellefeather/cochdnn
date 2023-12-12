import os
import scipy.io.wavfile as wav
import h5py
import pickle
import random
import numpy as np
from shutil import copyfile
import glob
import scipy
try:
    from scipy.misc import imread, imresize
except:
    pass
from functools import partial
import resampy
from PIL import Image
import json  
from analysis_scripts.default_paths import *

def generate_import_audio_functions(audio_func='psychophysicskell2018dry', preproc_scaled=1, rms_normalize=1, **kwargs):
  """
  Wrapper to choose which type of audio function to import.
  Input
  -----
  audio_func : a string determining which function will be returned to import the audio
  preproc_scaled (float) : multiplies the input audio by this value for scaling
  rms_normalize (None or float) : if not None, sets the RMS value to this float. 

  Returns
  -------
  audio_function : a function that takes in an index and returns a dictionary with (at minimum) the audio corresponding to the index along with the SR

  """
  if audio_func == 'psychophysicskell2018dry_overlap_jsinv3':
    return partial(psychophysicskell2018dry_overlap_jsinv3, preproc_scaled=preproc_scaled, rms_normalize=rms_normalize, **kwargs)
  elif audio_func == 'psychophysics_wsj400_jsintest':
    return partial(psychophysics_wsj400_jsintest, preproc_scaled=preproc_scaled, rms_normalize=rms_normalize, **kwargs)
  elif audio_func == 'load_specified_audio_path':
    return partial(use_audio_path_specified_audio, preproc_scaled=preproc_scaled, rms_normalize=rms_normalize, **kwargs)


def psychophysicskell2018dry_overlap_jsinv3(WAV_IDX, preproc_scaled=1, rms_normalize=None, SR=20000):
  """
  Loads an example from the dry psychophysics set used in kell2018 that is overlapped with the set in jsinv3
  This set contains 295 words

  Metamers from this set were used in Feather et al. 2019 (NeurIPS) 
  """

  # Contains ONLY the dry stimuli
  save_dry_path = os.path.join(ASSETS_PATH, 'behavioralKellDataset_sr20000_kellBehavioralDataset_jsinv3overlap_dry_only.pckl')

  with open(save_dry_path, 'rb') as handle:
    behavioral_dataset_kell = pickle.load(handle)

  word_to_int = dict(zip(behavioral_dataset_kell['stimuli']['word'], behavioral_dataset_kell['stimuli']['word_int']))
  int_to_word = dict(zip(behavioral_dataset_kell['stimuli']['word_int'], behavioral_dataset_kell['stimuli']['word']))
    
  SR_loaded = behavioral_dataset_kell['stimuli']['sr'][WAV_IDX]
  wav_f = behavioral_dataset_kell['stimuli']['signal'][WAV_IDX]
  if SR_loaded != SR:
    print('RESAMPLING')
    wav_f = resampy.resample(wav_f, SR_loaded, SR)
    SR_loaded = SR

  wav_f = wav_f * preproc_scaled # some of networks require us to scale the audio

  print("Loading: %s"%behavioral_dataset_kell['stimuli']['source'][WAV_IDX])
  
  if rms_normalize is not None:
    wav_f = wav_f - np.mean(wav_f.ravel())
    wav_f = wav_f/(np.sqrt(np.mean(wav_f.ravel()**2)))*rms_normalize
    print(np.sqrt(np.mean(wav_f.ravel()**2)))
    rms = rms_normalize
  else:
    rms = b['stimuli']['rms'][WAV_IDX]
  audio_dict={}

  audio_dict['wav'] = wav_f
  audio_dict['SR'] = SR
  audio_dict['word_int'] = behavioral_dataset_kell['stimuli']['word_int'][WAV_IDX]
  audio_dict['word'] = behavioral_dataset_kell['stimuli']['word'][WAV_IDX]
  audio_dict['rms'] = rms
  audio_dict['filename'] = behavioral_dataset_kell['stimuli']['path'][WAV_IDX]
  audio_dict['filename_short'] = behavioral_dataset_kell['stimuli']['source'][WAV_IDX]
  audio_dict['correct_response'] = behavioral_dataset_kell['stimuli']['word'][WAV_IDX]

  return audio_dict


def psychophysics_wsj400_jsintest(WAV_IDX, preproc_scaled=1, rms_normalize=None, SR=20000):
  """
  Loads an example from a set of 400 WSJ clips pulled from the jsinv3 test set.
  Each clip is of a different word, and a unique clip from WSJ (ie no two clips that are back to back words)

  Metamers from this set were used in Feather et al. 2022
  """

  pckl_path = os.path.join(ASSETS_PATH, 'word_WSJ_validation_jsin_400words_1samplesperword_with_metadata.pckl')
  with open(pckl_path, 'rb') as handle:
    behavioral_dataset = pickle.load(handle)
    
  word = behavioral_dataset['Dataset_Word_Order'][WAV_IDX]
  word_data = behavioral_dataset[word]
  assert word_data['dataframe_metadata']['word'] == word

  SR_loaded = word_data['dataframe_metadata']['sr']
  wav_f = word_data['audio_clips'][0]

  if SR_loaded != SR:
    print('RESAMPLING')
    wav_f = resampy.resample(wav_f, SR_loaded, SR)
    SR_loaded = SR

  wav_f = wav_f * preproc_scaled # some of networks require us to scale the audio

  print("Loading: %s"%word)

  # Always mean subtract the clip in this dataset. 
  wav_f = wav_f - np.mean(wav_f.ravel())
  rms_clip = np.sqrt(np.mean(wav_f.ravel()**2))

  if rms_normalize is not None:
    wav_f = wav_f/rms_clip*rms_normalize
    rms = rms_normalize
  else:
    rms = rms_clip
    
  audio_dict={}

  old_path = word_data['dataframe_metadata']['path']
  path_without_root = old_path.split('/home/raygon/projects/user/jfeather/')[-1]
    
  audio_dict['wav'] = wav_f
  audio_dict['SR'] = SR
  audio_dict['word_int'] = word_data['dataframe_metadata']['word_int']
  audio_dict['word'] = word_data['dataframe_metadata']['word']
  audio_dict['rms'] = rms
  audio_dict['filename'] = path_without_root
  audio_dict['filename_short'] = word_data['dataframe_metadata']['source']
  audio_dict['correct_response'] =  word_data['dataframe_metadata']['word']

  return audio_dict


def use_audio_path_specified_audio(WAV_IDX, wav_path=None, wav_word=None, 
                                   preproc_scaled=1, rms_normalize=None, SR=20000):
  """
  Loads an example wav specified by wav_path
  """
  del WAV_IDX

  word_and_speaker_encodings = pickle.load( open(WORD_AND_SPEAKER_ENCODINGS_PATH, "rb" ))
  word_to_int = word_and_speaker_encodings['word_to_idx']

  print("Loading: %s"%wav_path)
  SR_loaded, wav_f = scipy.io.wavfile.read(wav_path)
  if SR_loaded != SR:
    wav_f = resampy.resample(wav_f, SR_loaded, SR)
    SR_loaded = SR

  if rms_normalize is not None:
    wav_f = wav_f - np.mean(wav_f.ravel())
    wav_f = wav_f/(np.sqrt(np.mean(wav_f.ravel()**2)))*rms_normalize
    rms = rms_normalize
  else:
    rms = np.sqrt(np.mean(wav_f.ravel()**2))

  wav_f = wav_f * preproc_scaled # some of networks require us to scale the audio

  audio_dict={}

  audio_dict['wav'] = wav_f
  audio_dict['SR'] = SR
  audio_dict['word_int'] = word_to_int[wav_word]
  audio_dict['word'] = wav_word
  audio_dict['rms'] = rms
  audio_dict['filename'] = wav_path
  audio_dict['filename_short'] = wav_path.split('/')[-1]
  audio_dict['correct_response'] = wav_word

  return audio_dict


def use_pytorch_datasets(IMG_IDX, DATA, train_or_val='val', im_shape=224, data_format='NHWC'):
  print('Using pytorch dataset %s'%DATA)
  if im_shape !=224:
    raise NotImplementedError('Pytorch datasets not implemented for arbitrary shapes yet')

  from robustness import datasets

  DATA_PATH_DICT = { # Add additional datasets here if you want. 
      'ImageNet': IMAGENET_PATH, 
  } 

  BATCH_SIZE = 1
  if train_or_val == 'train':
    raise NotImplementedError('train subset is not implemented for pytorch datasets')
  elif train_or_val == 'val':
    only_val = True
  else:
    raise ValueError("train_or_val must be 'train' or 'val', currently set as %s"%train_or_val)

  dataset_function = getattr(datasets, DATA)
  dataset = dataset_function(DATA_PATH_DICT[DATA])
  train_loader, test_loader = dataset.make_loaders(workers=0, 
                                                   batch_size=BATCH_SIZE, 
                                                   data_aug=False,
                                                   subset_val=1,
                                                   subset_start=IMG_IDX,
                                                   shuffle_val=False,
                                                   only_val=only_val)
  data_iterator = enumerate(test_loader)
  _, (im, targ) = next(data_iterator) # Images to invert

  if data_format=='NCHW':
    im = np.array(im)
  elif data_format=='NHWC':
    im = np.rollaxis(np.array(im),1,4)
  else:
    raise ValueError('Unsupported data_format %s'%data_format)

  image_dict = {}
  image_dict['image'] = im
  image_dict['shape'] = im_shape
  image_dict['filename'] = 'pytorch_%s_%s_IMG_IDX'%(DATA, train_or_val)
  image_dict['filename_short'] = 'pytorch_%s_%s_IMG_IDX'%(DATA, train_or_val)
  image_dict['correct_response'] = targ
  image_dict['max_value_image_set'] = 1
  image_dict['min_value_image_set'] = 0
  return image_dict


def get_multiple_samples_pytorch_datasets(NUM_EXAMPLES, DATA, train_or_val='val', im_shape=224, data_format='NHWC', START_IDX=0):
  print('Using pytorch dataset %s'%DATA)
  if im_shape !=224:
    raise NotImplementedError('Pytorch datasets not implemented for arbitrary shapes yet')
  # This uses the robustness code to load the datasets in
  from robustness import datasets

  DATA_PATH_DICT = { # Add additional datasets here if you want.
      'ImageNet': IMAGENET_PATH,
  }

  BATCH_SIZE = 1
  if train_or_val == 'train':
    raise NotImplementedError('train subset is not implemented for pytorch datasets')
  elif train_or_val == 'val':
    only_val = True
  else:
    raise ValueError("train_or_val must be 'train' or 'val', currently set as %s"%train_or_val)

  dataset_function = getattr(datasets, DATA)
  dataset = dataset_function(DATA_PATH_DICT[DATA])

  train_loader, test_loader = dataset.make_loaders(workers=0,
                                                   batch_size=BATCH_SIZE,
                                                   data_aug=False,
                                                   subset_start=START_IDX,
                                                   shuffle_val=True,
                                                   only_val=only_val)
  data_iterator = enumerate(test_loader)
  _, (im, targ) = next(data_iterator) # Images to invert

  all_images = []
  correct_response = []
  for IMG_IDX in range(NUM_EXAMPLES):
    _, (im, targ) = next(data_iterator)
    if data_format=='NCHW':
      im = np.array(im)
    elif data_format=='NHWC':
      im = np.rollaxis(np.array(im),1,4)
    else:
      raise ValueError('Unsupported data_format %s'%data_format)
    all_images.append(im)
    correct_response.append(np.array(targ))

  image_dict = {}
  image_dict['shape'] = im_shape
  image_dict['max_value_image_set'] = 1
  image_dict['min_value_image_set'] = 0
  return all_images, correct_response, image_dict

def read_sound_file_list(filepath, remove_extension=False):
    """
    Takes in a text file with one sound on each line. Returns a python list with one element for each line of the file. Option to remove the extension at the end of the filename.

    Inputs
    ------
    filepath : string
        The path to the text file containing a list of sounds
    remove_exnention : Boolean
        If true, removes the extension (if it exists) from the list of sounds.

    Returns
    -------
    all_sounds : list
        The sounds within filepath as a list.

    """
    with open(filepath,'r') as f:
        all_sounds = f.read().splitlines()
    if remove_extension:
        for sound_idx, sound in enumerate(all_sounds):
            all_sounds[sound_idx] = sound.split('.')[0]
    return all_sounds
