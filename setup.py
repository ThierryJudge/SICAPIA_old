#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='sicapia',
      version='0.0.1',
      description='Describe Your Cool Project',
      author='',
      author_email='',
      url='https://github.com/williamFalcon/pytorch-lightning-conference-seed',  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
      install_requires=[
            'pytorch-lightning', 'torchvision', 'matplotlib', 'numpy', 'torch', 'scipy'
      ],
      packages=find_packages()
      )

