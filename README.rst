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
  git pull https://github.com/andrisecker/assemblyfire.git
  cd assemblyfire/assemblyfire
  pip install -i https://bbpteam.epfl.ch/repository/devpi/simple -e .[bluepy]

  # to install it locally without having to install bluepy
  git pull https://github.com/andrisecker/assemblyfire.git
  cd assemblyfire/assemblyfire
  pip install -e .


Examples
--------

.. code-block::

  assemblyfire find-assemblies -v configs/100p_depol_simmat.yaml
  assemblyfire consenus -v configs/100p_depol_simmat.yaml
  assemblyfire conn-mat -v configs/100p_depol_simmat.yaml
  assemblyfire single-cell -v configs/100p_depol_simmat.yaml
