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
  cd assemblyfire/
  pip install -i https://bbpteam.epfl.ch/repository/devpi/simple -e .[bluepy]

  # to install it locally without having to install bluepy
  git pull https://github.com/andrisecker/assemblyfire.git
  cd assemblyfire/
  pip install -e .


Examples
--------

.. code-block::

  assemblyfire find_assemblies -v configs/100p_depol_simmat.yaml
  assemblyfire consenus -v configs/100p_depol_simmat.yaml
