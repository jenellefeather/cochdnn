import requests 
import tarfile
import sys
import os
from default_paths import * 

def download_extract_remove(url, extract_location):
    temp_file_location = os.path.join(extract_location, 'temp.tar')
    print('Downloading %s to %s'%(url, temp_file_location))
    with open(temp_file_location, 'wb') as f:
        r = requests.get(url, stream=True)
        for chunk in r.raw.stream(1024, decode_content=False):
            if chunk:
                f.write(chunk)
                f.flush()
    print('Extracting %s'%temp_file_location)
    tar = tarfile.open(temp_file_location)
    tar.extractall(path=extract_location) # untar file into same directory
    tar.close()

    print('Removing temp file %s'%temp_file_location)
    os.remove(temp_file_location)

# Download the model checkpoints (~14GB)
url_cochdnn_checkpoints = 'https://mcdermottlab.mit.edu/cochdnn/cochdnn_model_checkpoints.tar'
download_extract_remove(url_cochdnn_checkpoints, '')
