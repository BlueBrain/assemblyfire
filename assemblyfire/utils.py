# -*- coding: utf-8 -*-
"""
Assembly detection related utility functions
(mostly loading simulation related stuff)
author: András Ecker, last update: 11.2020
"""

import os
import json
import numpy as np
from bluepy.v2.enums import Cell, Synapse
from bluepy.v2 import Circuit


def ensure_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def get_out_fname(root_path, clustering_method):
    if clustering_method == "hierarchical":
        tmp = "assemblies_simmat.h5"
    elif clustering_method == "density_based":
        tmp = "assemblies_spikes.h5"
    return os.path.join(root_path, tmp)


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
    return c.cells.ids({"$target":target})


def get_E_gids(c, target="mc2_Column"):
    return c.cells.ids({"$target": target, Cell.SYNAPSE_CLASS: "EXC"})


def get_EI_gids(c, target="mc2_Column"):
    gidsE = get_E_gids(c, target)
    gidsI = c.cells.ids({"$target": target, Cell.SYNAPSE_CLASS: "INH"})
    return gidsE, gidsI


def _get_layer_gids(c, layer, target):
    return c.cells.ids({"$target": target, Cell.LAYER: layer})


def map_gids_to_depth(circuit_config, target="mc2_Column"):
    """Creates gid-depth map (for better figure asthetics)"""

    c = Circuit(circuit_config)
    gids = _get_gids(c, target)
    ys = c.cells.get(gids)["y"]
    # convert pd.Series to dictionary...
    gids = np.asarray(ys.index)
    depths = ys.values
    return {gid: depths[i] for i, gid in enumerate(gids)}


def get_layer_boundaries(circuit_config, target="mc2_Column"):
    """Gets layer boundaries and cell numbers (used for raster plots)"""

    c = Circuit(circuit_config)
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