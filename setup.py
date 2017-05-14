#!/usr/bin/env python
"""
2017-05-14 19:45:08
@author: Paul Reiter
"""
from setuptools import setup

setup(name='pywbm',
      version='0.1',
      description='A python implementation of the wave based method for'
                  'acoustics.',
      author='Paul Reiter',
      author_email='reiter.paul@gmail.com',
      packages=['pywbm'],
      zip_safe=False,
      install_requires=['numpy', 'scipy', 'matplotlib'],
      tests_require=['pytest'])
