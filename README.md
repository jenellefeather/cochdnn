# CochDNN
Model code for CochDNN auditory models (auditory models with cochleagram front end). 

Snippet for loading a model using a build script: 
```
import os
import importlib

# Choose the model that will be loaded
model_dir = 'kell2018_audioset_decay_lr'

build_network_spec = importlib.util.spec_from_file_location("build_network",
                        os.path.join(self.model_directory_base, model_dir, 'build_network.py'))
build_network = importlib.util.module_from_spec(build_network_spec)
build_network_spec.loader.exec_module(build_network)

model, ds = build_network.main()
```
