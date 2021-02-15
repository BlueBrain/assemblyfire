# -*- coding: utf-8 -*-
"""

last modified: András Ecker 02.2021
"""

import os
import pickle
from collections import namedtuple
import numpy as np
from assemblyfire.spikes import spikes2mat, sign_rate_std, calc_rate, spikes_to_h5
from assemblyfire.clustering import cluster_spikes, detect_assemblies
from assemblyfire.plots import plot_rate

SpikeMatrixResult = namedtuple("SpikeMatrixResult", ["spike_matrix", "gids", "t_bins"])
FigureArgs = namedtuple("FigureArgs", ["stim_times", "patterns", "depths", "ystuff", "fig_path"])
data_dir = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/assemblyfire/use_cases/input_data"
fig_path = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/figures/toposample"


def load_spikes(data_dir):
    """Loads spikes from .npy file (only EXC spikes are saved)"""
    data = np.load(os.path.join(data_dir, "raw_spikes.npy"))
    return data[:, 0], data[:, 1]


def load_patterns(data_dir):
    """Loads stimulus stream (AKA patterns) from .npy file and converts numbers to letters as pattern ids"""
    data = np.load(os.path.join(data_dir, "stim_stream.npy"))
    stim_times = np.arange(len(data)) * 200  # 1 randomly selected pattern is presented in every 200 ms
    patterns = np.empty(data.shape, dtype=object)
    for pattern_id, pattern_name in zip(np.arange(8), ["A", "B", "C", "D", "E", "F", "G", "H"]):
        patterns[data == pattern_id] = pattern_name
    return stim_times.tolist(), patterns.tolist()


def map_gids_to_depth(data_dir):
    """Rewrite of `assemblyfire/utils.py/map_gids_to_depth() using only the saved pickle file"""
    with open(os.path.join(data_dir, "neuron_info.pickle"), "rb") as f:
        data = pickle.load(f)
    nrn_info = data[data["synapse_class"] == "EXC"]  # filter out INH cells (we don't need them here)
    gids = np.asarray(nrn_info.index)
    depths = nrn_info["y"].values
    return {gid: depths[i] for i, gid in enumerate(gids)}


def get_layer_boundaries(data_dir):
    """Rewrite of `assemblyfire/utils.py/get_layer_boundaries() using only the saved pickle file"""
    with open(os.path.join(data_dir, "neuron_info.pickle"), "rb") as f:
        data = pickle.load(f)
    yticks = []
    yticklables = []
    hlines = []
    for layer in range(1, 7):
        nrn_info = data[data["layer"] == layer]
        yticklables.append("L%i\n(%i)" % (layer, len(nrn_info)))
        ys = nrn_info["y"]
        yticks.append(ys.mean())
        if layer == 1:
            hlines.append(ys.max())
            hlines.append(ys.min())
        else:
            hlines.append(ys.min())
    return {"yticks": yticks, "yticklabels": yticklables, "hlines": hlines}


def get_sign_spike_matrices(data_dir, h5f_name, fig_path, bin_size=10, seed=166273):
    """Quick and dirty rewrite of `assemblyfire/spikes.py/SpikeMatrixGroup.get_sign_spike_matrices()`
    for the use case of the topological sampling paper"""

    # load spikes and patterns
    spike_times, spiking_gids = load_spikes(data_dir)
    stim_times, patterns = load_patterns(data_dir)
    t_start = 0
    t_end = stim_times[-1] + 200
    spike_matrix, gids, t_bins = spikes2mat(spike_times, spiking_gids, t_start, t_end, bin_size)
    assert (spike_matrix.shape[0] == np.sum(spike_matrix.any(axis=1)))

    # threshold rate
    rate = np.sum(spike_matrix, axis=0)
    rate_th = sign_rate_std(spike_times, spiking_gids, t_start, t_end, bin_size, N=100)
    t_idx = np.where(rate > np.mean(rate) + rate_th)[0]
    rate_norm = len(np.unique(spiking_gids)) * 1e-3 * bin_size
    fig_name = os.path.join(fig_path, "rate_seed%i.png" % seed)
    plot_rate(rate / rate_norm, rate_th / rate_norm, t_start, t_end, fig_name)

    # store in the same format as one would for multiple seeds (just for consistency and code reuse)
    spike_matrix_dict = {seed: SpikeMatrixResult(spike_matrix[:, t_idx], gids, t_bins[t_idx])}
    # metadata = {"seeds": [seed], "stim_times": stim_times,
    #             "patterns": patterns}  # can't be saved as metadata as it's too large
    spikes_to_h5(h5f_name, spike_matrix_dict, metadata={}, prefix="spikes")

    return spike_matrix_dict, stim_times, patterns


if __name__ == "__main__":

    h5f_name = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/assemblyfire/use_cases/assemblies.h5"
    depths = map_gids_to_depth(data_dir)
    ystuff = get_layer_boundaries(data_dir)
    spike_matrix_dict, stim_times, patterns = get_sign_spike_matrices(data_dir, h5f_name, fig_path)
    clusters_dict = cluster_spikes(spike_matrix_dict, method="hierarchical",
                                   FigureArgs=FigureArgs(stim_times, patterns, depths, None, fig_path))
    detect_assemblies(spike_matrix_dict, clusters_dict, h5f_name, h5_prefix="assemblies",
                      FigureArgs=FigureArgs(None, None, depths, ystuff, fig_path))
