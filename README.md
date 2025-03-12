# CochDNN
Model code for loading CochDNN auditory models (auditory models with cochleagram front end). 

Contains the in-house models used in the paper:
Greta Tuckute*, Jenelle Feather*, Dana Boebinger, Josh H. McDermott (2023): _Many but not all deep neural network audio models capture brain responses and exhibit correspondence between model stages and brain regions_.

## Installation and downloading checkpoints. 
Required dependencies are specified in setup.py. Install with `pip install -e .`

Model checkpoints (~14GB) can be downloaded and extracted into the appropriate location with the included script: 
`python download_large_files.py`

To test if models load after installation and checkpoint downloading, you can run the test script `python tests/test_cochdnn.py`. Note: this script will attempt to load the model checkpoints into each architecture for the models without random permutations.

## Snippet for loading a model using a build script: 
```
import os
import importlib

# Choose the model that will be loaded
model_dir = 'resnet50_audioset'

build_network_spec = importlib.util.spec_from_file_location("build_network",
                        os.path.join(model_dir, 'build_network.py'))
build_network = importlib.util.module_from_spec(build_network_spec)
build_network_spec.loader.exec_module(build_network)

model, ds = build_network.main()
```
