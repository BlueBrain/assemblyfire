# -*- coding: utf-8 -*-
"""
Codebase adapted for the use case of the data from the Topological sampling paper (Reimann et al. 2020 bioRxiv)
(Spikes loaded from saved npz files instead of extracted with bluepy + the 15min long sim is sliced up to 30sec chunks)
last modified: András Ecker 02.2021
"""

import os
import pickle
from collections import namedtuple
import numpy as np
from assemblyfire.spikes import spikes2mat, spikes_to_h5  # sign_rate_std
from assemblyfire.clustering import cluster_sim_mat, detect_assemblies
from assemblyfire.utils import load_assemblies_from_h5
from assemblyfire.assemblies import consensus_over_seeds_hamming
from assemblyfire.plots import plot_rate, plot_sim_matrix,\
    plot_dendogram_silhouettes, plot_cluster_seqs, plot_pattern_clusters

SpikeMatrixResult = namedtuple("SpikeMatrixResult", ["spike_matrix", "gids", "t_bins"])
FigureArgs = namedtuple("FigureArgs", ["stim_times", "patterns", "depths", "ystuff", "fig_path"])
data_dir = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/assemblyfire/use_cases/input_data"
fig_path = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/figures/toposample"
# seed = 166273  # original random seed


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
    return stim_times, patterns


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


def get_sign_spike_matrices(data_dir, h5f_name, fig_path, bin_size=20):
    """Quick and dirty rewrite of `assemblyfire/spikes.py/SpikeMatrixGroup.get_sign_spike_matrices()`
    for the use case of the topological sampling paper (to loading spikes from file)"""

    spike_times, spiking_gids = load_spikes(data_dir)
    stim_times, patterns = load_patterns(data_dir)

    # slice up long sim (to 30sec long parts as it was simulated)
    t_full = stim_times[-1] + 200
    t_slices = np.arange(0, t_full+30000, 30000)
    spike_matrix_dict = {}
    for i, (t_start, t_end) in enumerate(zip(t_slices[:-1], t_slices[1:])):
        idx = np.where((t_start <= spike_times) & (spike_times < t_end))[0]
        spike_matrix, gids, t_bins = spikes2mat(spike_times[idx], spiking_gids[idx], t_start, t_end, bin_size)
        assert (spike_matrix.shape[0] == np.sum(spike_matrix.any(axis=1)))
        rate = np.sum(spike_matrix, axis=0)
        #rate_th = sign_rate_std(spike_times[idx], spiking_gids[idx], t_start, t_end, bin_size, N=100)
        rate_th = 0.  # don't threshold
        t_idx = np.where(rate > rate_th)[0]  # just std, not mean + std
        rate_norm = len(np.unique(spiking_gids[idx])) * 1e-3 * bin_size
        fig_name = os.path.join(fig_path, "rate_seed%i.png" % i)
        plot_rate(rate / rate_norm, rate_th / rate_norm, t_start, t_end, fig_name)
        # store in the same format as one would for multiple seeds (just for consistency and code reuse)
        spike_matrix_dict[i] = SpikeMatrixResult(spike_matrix[:, t_idx], gids, t_bins[t_idx])

    # metadata = {"stim_times": stim_times, "patterns": patterns}  # can't be saved as attr because it's too long
    spikes_to_h5(h5f_name, spike_matrix_dict, metadata={}, prefix="spikes")

    return spike_matrix_dict, stim_times, patterns, t_slices


def cluster_spikes(spike_matrix_dict, stim_times, patterns, t_slices, fig_path):
    """Quick and dirty rewrite of `assemblyfire/clustering.py/cluster_spikes()`
    for the use case of the topological sampling paper (to slice up patterns)"""

    clusters_dict = {}
    for i, (t_start, t_end) in enumerate(zip(t_slices[:-1], t_slices[1:])):
        idx = np.where((t_start <= stim_times) & (stim_times < t_end))[0]
        spike_matrix = spike_matrix_dict[i].spike_matrix
        t_bins = spike_matrix_dict[i].t_bins
        sim_matrix, clusters, plotting = cluster_sim_mat(spike_matrix)
        clusters_dict[i] = clusters

        fig_name = os.path.join(fig_path, "similarity_matrix_seed%i.png" % i)
        plot_sim_matrix(sim_matrix, t_bins, stim_times[idx], patterns[idx], fig_name)
        fig_name = os.path.join(fig_path, "Ward_clustering_seed%i.png" % i)
        plot_dendogram_silhouettes(clusters, *plotting, fig_name)
        fig_name = os.path.join(fig_path, "cluster_seq_seed%i.png" % i)
        plot_cluster_seqs(clusters, t_bins, stim_times[idx], patterns[idx], fig_name)
        fig_name = os.path.join(fig_path, "clusters_patterns_seed%i.png" % i)
        plot_pattern_clusters(clusters, t_bins, stim_times[idx], patterns[idx], fig_name)

    return clusters_dict


if __name__ == "__main__":

    h5f_name = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/assemblyfire/use_cases/assemblies.h5"
    spike_matrix_dict, stim_times, patterns, t_slices = get_sign_spike_matrices(data_dir, h5f_name, fig_path)
    clusters_dict = cluster_spikes(spike_matrix_dict, stim_times, patterns, t_slices, fig_path)
    depths = map_gids_to_depth(data_dir)
    ystuff = get_layer_boundaries(data_dir)
    detect_assemblies(spike_matrix_dict, clusters_dict, h5f_name, h5_prefix="assemblies",
                      FigureArgs=FigureArgs(None, None, depths, ystuff, fig_path))
    # load assemblies from file, create consensus assemblies and saving them to h5...
    assembly_grp_dict = load_assemblies_from_h5(h5f_name, prefix="assemblies", load_metadata=False)
    consensus_over_seeds_hamming(assembly_grp_dict, h5f_name,
                                 h5_prefix="consensus", fig_path=fig_path)
