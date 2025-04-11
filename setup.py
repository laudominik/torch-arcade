#!/usr/bin/env python
import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "torch_arcade",
    version = "1.0.0",
    author = "Dominik Lau",
    description = ("PyTorch ARCADE dataset."),
    packages = ['torch_arcade'],
    package_dir={'' : '.'},
    long_description=read('README.md'),
    install_requires=['torch', 'torchvision', 'pycocotools'],
)
