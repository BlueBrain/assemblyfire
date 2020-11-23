assemblyfire
============

find and analyze cell assemblies


Installation
------------

.. code-block::

  module purge
  module load unstable
  module load python
  python -m venv dev-assemblyfire
  source dev-assemblyfire/bin/activate
  git pull https://github.com/andrisecker/assemblyfire.git
  cd assemblyfire/
  pip install -i https://bbpteam.epfl.ch/repository/devpi/simple -e .


Examples
--------

.. code-block::

  assemblyfire find_assemblies -v /gpfs/bbp.cscs.ch/project/proj96/home/ecker/assemblyfire/configs/100p_depol_simmat.yaml
