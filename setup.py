#!/usr/bin/env python

from setuptools import setup

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
        "torch",
        "torchaudio",
        "h5py",
        "numpy",
        "dill",
        "cox",
        "tables",
        "tqdm",
        "resampy",
        "tensorboardX",
        "chcochleagram @ git+https://github.com/jenellefeather/chcochleagram.git"
]

setup(
    name='cochdnn',
    version='1.0.0',
    description="Models trained on word, speaker, noise task with cochleagram front ends.",
    long_description=readme,
    author="Jenelle Feather",
    author_email='jfeather@mit.edu',
    install_requires=requirements,
    py_modules=['tests','robustness','default_paths', 'analysis_scripts'],
    license="MIT license",
    keywords='audio, deep neural networks',
)
