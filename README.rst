assemblyfire
============

find and analyze cell assemblies


Installation
------------

module purge
module load unstable
module load python
python -m venv dev-assemblyfire
source dev-assemblyfire/bin/activate

git pull ...
cd assemblyfire/
pip install -i https://bbpteam.epfl.ch/repository/devpi/simple -e .


Examples
--------

assemblyfire find_assemblies "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/assemblyfire/configs/100p_depol_simmat.yaml"
