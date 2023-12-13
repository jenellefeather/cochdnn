"""
Contains tests for the public model metamers repository
"""

import unittest
import numpy as np
import torch as ch
import faulthandler
from pathlib import Path
import os
faulthandler.enable()
import imp
import importlib
import sys
import robustness

# For testing metamer generation. 
from analysis_scripts import * 

AUDIO_NETWORK_LIST = os.listdir('model_directories')

class AudioNetworkTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.base_directory = os.path.dirname(os.path.abspath(__file__))
        self.model_directory_base = os.path.join(self.base_directory,
                                                 'model_directories')

    def test_build_networks(self):
        for model in AUDIO_NETWORK_LIST:
            if 'randomize_weights' in model:
                continue
            with self.subTest(model=model):
                with ch.no_grad():
                    build_network_spec = importlib.util.spec_from_file_location("build_network",
                                            os.path.join(self.model_directory_base, model, 'build_network.py'))
                    build_network = importlib.util.module_from_spec(build_network_spec)
                    build_network_spec.loader.exec_module(build_network)
    
                    model, ds, metamer_layers = build_network.main(return_metamer_layers=True)
                    self.assertIsInstance(model, robustness.attacker.AttackerModel)
                    model.cpu()
                    del model 
                    del ds
                    del metamer_layers
                    ch.cuda.empty_cache()
                    del build_network
                    del build_network_spec

    def test_eval(self):
        for model in AUDIO_NETWORK_LIST[0:2]:
            with self.subTest(model=model):
               eval_natural_jsinv3.main(('-D model_directories/%s'%model).split())
                

if __name__ == "__main__":
    unittest.main()
    
