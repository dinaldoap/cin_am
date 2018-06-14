#!/usr/bin/env python

from distutils.core import setup

setup(name='KCM-K-GH',
      version='0.1',
      description='Implementação do KCM-K-GH',
      author='Cleison Amorim',
      author_email='amorim.cleison@gmail.com',
      install_requires=[
            'numpy >= 1.14.2', 
            'scipy >= 1.0.1',
            'sklearn >= 0.19.1'
            ]
     )