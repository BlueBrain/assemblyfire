assemblyfire
============

find and analyze cell assemblies


Installation
------------

.. code-block::

  # to install it on BB5 with bluepy:
  module purge
  module load unstable
  module load python
  python -m venv dev-assemblyfire
  source dev-assemblyfire/bin/activate
  git clone https://github.com/andrisecker/assemblyfire.git
  cd assemblyfire/
  pip install -i https://bbpteam.epfl.ch/repository/devpi/simple -e .[bluepy]
  # for some reason conntility needs to be installed separately (see setup.py for gitlab path)

  # to install it locally without having to install bluepy
  git clone https://github.com/andrisecker/assemblyfire.git
  cd assemblyfire/
  pip install -e .

  # flagser needs to be compiled and installed separately (see setup.py for github path)


Examples
--------

.. code-block::

  assemblyfire -v find-assemblies configs/v7_bbp-workflow.yaml
  assemblyfire -v consensus configs/v7_bbp-workflow.yaml
  assemblyfire -v conn-mat configs/v7_bbp-workflow.yaml
  assemblyfire -v single-cell configs/v7_bbp-workflow.yaml
