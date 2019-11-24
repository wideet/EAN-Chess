from setuptools import find_packages

#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

# getting absolute path for the project directory relative to setup.py file.
PROJECT_PATH = os.path.abspath(__file__ + '/..')

with open(os.path.join(PROJECT_PATH, 'requirements.txt')) as f:
    required = f.read().splitlines()

setup(
    name='ean_chess',
    version='0.1',
    author='Wideet Shende',
    author_email='wideet@gmail.com',
    description='Package to process cost models',
    url='http://github.com/wideet/EAN-Chess',
    packages=find_packages(),
    install_requires=required,
    include_package_data=True,
)
