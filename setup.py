#!/usr/bin/env python

import imp
import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 8):
    sys.exit("Sorry, Python < 3.8 is not supported")

VERSION = imp.load_source("", "assemblyfire/version.py").__version__

setup(
    name="assemblyfire",
    authors=["Andras Ecker", "Michael Reimann", "Daniela Egas Santander"],
    author_email="andras.ecker@epfl.ch",
    version=VERSION,
    description="find and analyze cell assemblies",
    long_description=\
        """assemblyfire
           ============

           find and analyze cell assemblies
        """,
    long_description_content_type="text/x-rst",
    url="http://bluebrain.epfl.ch",
    license="LGPL-3.0",
    install_requires=["h5py>=3.7.0",
                      "pyyaml>=6.0",
                      "tqdm>=4.64.1",
                      "click>=8.1.3",
                      "cached-property>=1.5.2",
                      "numpy>=1.24.1",
                      "scipy>=1.10.0",
                      "pandas>=1.5.3",
                      "scikit-learn",
                      "libsonata>=0.1.16",
                      "pyflagser>=0.4.5",
                      "pyflagsercount>=0.2.2",
                      "matplotlib>=3.6.3",
                      "seaborn>=0.12.2",
                      "neurom>=3.2.2"],
    packages=find_packages(),
    python_requires=">=3.8",
    entry_points={"console_scripts": ["assemblyfire=assemblyfire.cli:cli"]},
    extras_require={
        "docs": ["sphinx", "sphinx-bluebrain-theme"],
        "bluepy": ["bluepy[all]>=2.4.6"],
        "conntility": ["Connectome-utilities @ https://bbpgitlab.epfl.ch/conn/structural/Connectome-utilities"]
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
