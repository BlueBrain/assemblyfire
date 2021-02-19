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
  pip install --upgrade cmake
  git clone --recursive https://github.com/andrisecker/assemblyfire.git
  cd assemblyfire/assemblyfire
  pip install -i https://bbpteam.epfl.ch/repository/devpi/simple -e .[bluepy]
  cd ../pyflagsercontain
  pip install -e .

  # to install it locally without having to install bluepy
  git clone --recursive https://github.com/andrisecker/assemblyfire.git
  cd assemblyfire/assemblyfire
  pip install -e .
  cd ../pyflagsercontain
  pip install -e .


Examples
--------

.. code-block::

  assemblyfire -v find-assemblies configs/100p_depol_simmat.yaml
  assemblyfire -v consensus configs/100p_depol_simmat.yaml
  assemblyfire -v conn-mat configs/100p_depol_simmat.yaml
  assemblyfire -v single-cell configs/100p_depol_simmat.yaml
