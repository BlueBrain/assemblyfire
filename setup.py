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
    install_requires=["h5py>=3.8.0",
                      "pyyaml>=6.0",
                      "tqdm>=4.64.1",
                      "click>=8.1.3",
                      "cached-property>=1.5.2",
                      "numpy>=1.24.3",
                      "scipy>=1.10.1",
                      "pandas>=2.0.2",
                      "scikit-learn<=0.24",  # pyitlib cannot handle later versions...
                      "pyitlib>=0.2.3",
                      "libsonata>=0.1.21",
                      "neurom>=3.2.2",
                      "morph-tool>=2.9.1",
                      "bluepysnap>=1.0.5",
                      "ConnectomeUtilities @ git+https://github.com/BlueBrain/ConnectomeUtilities",
                      "pyflagser>=0.4.5",
                      "pyflagsercount>=0.3.3",
                      "matplotlib>=3.7.1",
                      "seaborn>=0.12.2"],
    packages=find_packages(),
    python_requires=">=3.8",
    entry_points={"console_scripts": ["assemblyfire=assemblyfire.cli:cli"]},
    extras_require={
        "docs": ["sphinx", "sphinx-bluebrain-theme"],
        "bglibpy": ["bglibpy>=4.9.10"]  # this should be replaced once open sourced as `bluecellulab`
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
