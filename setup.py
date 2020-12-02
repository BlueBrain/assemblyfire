#!/usr/bin/env python

import imp
import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported")

with open("README.rst", encoding="utf-8") as f:
    README = f.read()

VERSION = imp.load_source("", "assemblyfire/version.py").__version__

setup(
    name="assemblyfire",
    author=["Andras Ecker", "Michael Reimann", "Daniela Egas Santander", "Nicolas Ninin"],
    author_email="andras.ecker@epfl.ch",
    version=VERSION,
    description="find and analyze cell assemblies",
    long_description=README,
    long_description_content_type="text/x-rst",
    url="http://bluebrain.epfl.ch",
    license="LGPL-3.0",
    install_requires=["h5py>=2.0.0,<3.0",
                      "pyyaml>=5.3.1",
                      "tqdm>=4.52.0",
                      "click>=7.1.2",
                      "cached-property>=1.5.2",
                      "numpy>=1.19.4",
                      "scipy>=1.5.4",
                      "pandas>=1.1.4",
                      "scikit-learn>=0.23.2",
                      "networkx>=2.5",
                      "scikit-network>=0.20.0",
                      "pyflagser>=0.4.2",
                      "matplotlib>=3.1.3",
                      "seaborn>=0.11.0",
                     ],
    packages=find_packages(),
    python_requires=">=3.6",
    entry_points={"console_scripts": ["assemblyfire=assemblyfire.cli:cli"]},
    extras_require={
        "docs": ["sphinx", "sphinx-bluebrain-theme"],
        "bluepy": ["bluepy[all]>=0.14.15"]
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
