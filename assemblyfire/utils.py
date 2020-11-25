# -*- coding: utf-8 -*-
"""
Assembly detection related utility functions
(mostly loading simulation related stuff)
author: András Ecker, last update: 11.2020
"""

import os
import json
import h5py
from collections import namedtuple
import numpy as np

SpikeMatrixResult = namedtuple("SpikeMatrixResult", ["spike_matrix", "gids", "t_bins"])


def _get_bluepy_circuit(circuitconfig_path):
    try:
        from bluepy.v2 import Circuit
    except ImportError as e:
        msg = (
            "Assemblyfire requirements are not installed.\n"
            "Please pip install bluepy as follows:\n"
            " pip install -i https://bbpteam.epfl.ch/repository/devpi/simple bluepy[all]"
        )
        raise ImportError(str(e) + "\n\n" + msg)
    return Circuit(circuitconfig_path)


def ensure_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def get_seeds(root_path):
    """Reads sim seeds from simwriter generated file"""
    f_name = os.path.join(root_path, "project_simulations.txt")
    with open(f_name, "r") as f:
        seeds = [int(l.strip().split('/')[-1][4:]) for l in f]
    return seeds


def get_patterns(root_path):
    """Return list of patterns presented during the simulation"""

    # get spike train name from json
    proj_name = root_path.split('/')[-1]
    jf_name = os.path.join(root_path, "%s.json"%proj_name)
    with open(jf_name, "rb") as f:
        configs = json.load(f)
        for param in configs["project_parameters"]:
            if param["name"] == "stimulus":
                spike_train_dir = param["kwargs"]["spike_train_dir"]
    spike_train_name = spike_train_dir.split('/')[-2]

    # load in pattern order txt saved by `spikewriter.py`
    patterns = []
    f_name = os.path.join("/gpfs/bbp.cscs.ch/project/proj96/home/ecker/simulations/spiketrains", "%s.txt"%spike_train_name)
    with open(f_name, "r") as f:
        for l in f:
            patterns.append(l.strip())
    return patterns


def _get_gids(c, target):
    import warnings
    warnings.simplefilter(action="ignore", category=FutureWarning)
    return c.cells.ids({"$target": target})


def get_E_gids(c, target="mc2_Column"):
    from bluepy.v2.enums import Cell
    import warnings
    warnings.simplefilter(action="ignore", category=FutureWarning)
    return c.cells.ids({"$target": target, Cell.SYNAPSE_CLASS: "EXC"})


def get_EI_gids(c, target="mc2_Column"):
    from bluepy.v2.enums import Cell
    import warnings
    warnings.simplefilter(action="ignore", category=FutureWarning)
    gidsE = get_E_gids(c, target)
    gidsI = c.cells.ids({"$target": target, Cell.SYNAPSE_CLASS: "INH"})
    return gidsE, gidsI


def _get_layer_gids(c, layer, target):
    from bluepy.v2.enums import Cell
    import warnings
    warnings.simplefilter(action="ignore", category=FutureWarning)
    return c.cells.ids({"$target": target, Cell.LAYER: layer})


def map_gids_to_depth(circuit_config, target="mc2_Column"):
    """Creates gid-depth map (for better figure asthetics)"""

    c = _get_bluepy_circuit(circuit_config)
    gids = _get_gids(c, target)
    ys = c.cells.get(gids)["y"]
    # convert pd.Series to dictionary...
    gids = np.asarray(ys.index)
    depths = ys.values
    return {gid: depths[i] for i, gid in enumerate(gids)}


def get_layer_boundaries(circuit_config, target="mc2_Column"):
    """Gets layer boundaries and cell numbers (used for raster plots)"""

    c = _get_bluepy_circuit(circuit_config)
    yticks = []
    yticklables = []
    hlines = []
    for layer in range(1, 7):
        gids = _get_layer_gids(c, layer, target)
        yticklables.append("L%i\n(%i)" % (layer, len(gids)))
        ys = c.cells.get(gids)["y"]
        yticks.append(ys.mean())
        if layer == 1:
            hlines.append(ys.max())
            hlines.append(ys.min())
        else:
            hlines.append(ys.min())
    return {"yticks": yticks, "yticklabels": yticklables, "hlines": hlines}


def get_spikes(sim, gids, t_start, t_end):
    """Extracts spikes (using bluepy)"""

    if gids is None:
        spikes = sim.spikes.get(t_start=t_start, t_end=t_end)
    else:
        spikes = sim.spikes.get(gids, t_start=t_start, t_end=t_end)
    return np.asarray(spikes.index), np.asarray(spikes.values)


def read_spikes(f_name, t_start, t_end):
    """Reads spikes from file"""

    ts = []
    gids = []
    with open(f_name, "rb") as f:
        next(f)  # skip "\scatter"
        for line in f:
            tmp = line.split()
            ts.append(float(tmp[0]))
            gids.append(int(tmp[1]))
    spike_times = np.asarray(ts)
    spiking_gids = np.asarray(gids)

    idx = np.where((t_start < spike_times) & (spike_times < t_end))[0]

    return spike_times[idx], spiking_gids[idx]


def load_assemblies_from_h5(h5f_name, prefix="assemblies"):
    """Load assemblies over seeds from saved h5 file into dict of AssemblyGroups"""
    from assemblyfire.assemblies import AssemblyGroup, AssemblyProjectMetadata

    with h5py.File(h5f_name, "r") as h5f:
        keys = list(h5f[prefix].keys())
    project_metadata = AssemblyProjectMetadata.from_h5(h5f_name, prefix="spikes")
    return dict([(k, AssemblyGroup.from_h5(h5f_name, k, prefix=prefix))
                 for k in keys]), project_metadata


def load_spikes_from_h5(h5f_name, prefix="spikes"):
    """Load spike matrices over seeds from saved h5 file"""
    from assemblyfire.assemblies import AssemblyProjectMetadata

    h5f = h5py.File(h5f_name, "r")
    seeds = list(h5f[prefix].keys())
    prefix_grp = h5f[prefix]
    spike_matrix_dict = {}
    for seed in seeds:
        spike_matrix_dict[seed] = SpikeMatrixResult(prefix_grp[seed]["spike_matrix"][:],
                                                    prefix_grp[seed]["gids"][:],
                                                    prefix_grp[seed]["t_bins"][:])
    h5f.close()
    project_metadata = AssemblyProjectMetadata.from_h5(h5f_name, prefix=prefix)
    return spike_matrix_dict, project_metadata