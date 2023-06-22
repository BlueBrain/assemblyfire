# Assemblyfire

Detection of cell assemblies, and analysis of their connectome.


## Examples

```
assemblyfire -v assemblies configs/np_10seeds.yaml  # find assemblies (for every repetition)
assemblyfire -v consensus configs/np_10seeds.yaml  # find "consensus" assemblies (across repetitions)
assemblyfire -v conn-mat configs/np_10seeds.yaml  # extract connectome from SONATA circuit
assemblyfire -v syn-nnd configs/np_10seeds.yaml seed19  # get normalized synapses nearest neighbour distances (in a given repetition)
assemblyfire -v single-cell configs/v7_10seeds.yaml  # calculate spike time reliability (across repetitions)
```

Once all the features are saved into a single HDF5 file, one can just simply load them, e.g. with the snippet below:
```
from assemblyfire.config import Config
from assemblyfire.utils import load_assemblies_from_h5
config = Config("configs/np_10seeds.yaml")
assembly_grp_dict, _ = load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)
for assembly in assembly_grp["seed19"].assemblies:
    assembly_neurons = assembly.gids
```
or do more involved analyses:
```
cd analysis_src
python assembly_topology.py  # diverse topological analyses
python compare_assemblies.py  # compare assemblies (within one config, or across configs)
python consensus_botany.py  # features of "consensus" assemblies, and their correlation to spike time reliability
python scan_nclusters.py  # detect different number of clusters of time bins (and corresponding assemblies)
```
To download the assemblies linked to `configs/np_10seeds.yaml` and used in our manuscript (see BibTeX below) use the following Zenodo link:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8052722.svg)](https://doi.org/10.5281/zenodo.8052722)


## Setting up one's own pipeline

The connectivity analysis part of the package is built to work with the SONATA format ([PLoS manuscript](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007696), [GitHub page](https://github.com/AllenInstitute/sonata/tree/master)) and relies on [snap](https://github.com/BlueBrain/snap) to do so. On the other hand, the cell assembly detection part is based purely on spike times (no information about the connectome is required) and thus can be used as a standalone. To do so one has to set up a new `.yaml` config specifying at least:

```
root_path: "<PATH>"

input_sequence_fname: "<FILE_NAME.txt>"

h5_out:
    file_name: "<FILE_NAME>.h5"
    prefixes:
        spikes: "<GROUP_NAME_WITHIN_HDF5_FILE>"
        assemblies: "<GROUP_NAME_WITHIN_HDF5_FILE>"

root_fig_path: "<PATH>"

preprocessing_protocol:
    node_pop: "NAME_OF_NODE_POPULATION_IN_SPIKE_REPORT"  # see below
    target: "NAME_OF_TARGET_IN_SONATA_NODE_SET"  # can be an empty string as well
    t_start:   # in (ms)
    t_end:  # in (ms)
    bin_size:  # in (ms)
```

Atm. everything is hard coded to our simulation pipeline, thus `assemblyfire` will look for a pickle file under the root path + `analysis/simulations.pkl` and will try to load it as a `pandas.Series` with `seed` as the name of the index. To change this to your liking, please update `utils.py/get_sim_path()`.

- If the values in the `Series` are paths to `.json` files, `assemblyfire` will assume those to be valid SONATA simulation configs, and try to load them as a `bluepysnap.Simulation` object, and extract spikes. 
- If the values in the `Series` are paths to `.h5` files, `assemblyfire` will assume those to be valid SONATA spike reports, and  try to load them as a `libsonata.SpikeReader` object, and extract spikes. (For an example for saving spikes in this format, see `tests/gen_test_spikes.py`).
- If you want to read spikes from an other format, just extend `spikes.py/load_spikes()`. They should be returned as 2 `np.array`s, the first one specifying the times of the spikes, the second one the IDs of the spiking neurons.


## Installation
Simply run `pip install .`

All dependencies are declared in the `setup.py` and (except [ConnectomeUtilities](https://github.com/BlueBrain/ConnectomeUtilities)) are available from [pypi](https://pypi.org/)


## Citation
If you use this software, kindly use the following BibTeX entry for citation:

```
@article{Ecker2023,
author = {Ecker, Andr{\'{a}}s and Santander, Daniela Egas and Bola{\~{n}}os-Puchet, Sirio and Isbister, James B. and Reimann, Michael W.},
doi = {https://doi.org/10.1101/2023.02.24.529863},
journal = {bioRxiv},
title = {{Cortical cell assemblies and their underlying connectivity: an in silico study}},
year = {2023}
}
```


## Acknowledgements & Funding
The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

Copyright (c) 2023 Blue Brain Project / EPFL.