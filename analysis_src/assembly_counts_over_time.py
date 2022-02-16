"""
Loads metadata and plots number of unique assemblies for every pattern presented through time
last modified: András Ecker 01.2022
"""

import os
import numpy as np

import assemblyfire.utils as utils
from assemblyfire.config import Config
from assemblyfire.plots import plot_n_assemblies


def count_assemblies(config_path):
    """Reloads significant time bins and cluster seqs from saved h5 file and counts the number of unique assemblies
    for every pattern presentation"""

    config = Config(config_path)
    spike_matrices, project_metadata = utils.load_spikes_from_h5(config.h5f_name, config.h5_prefix_spikes)
    stim_times, patterns = project_metadata["stim_times"], project_metadata["patterns"]
    t_chunks = project_metadata["t"]
    if len(t_chunks) == 2:
        raise RuntimeError("This script only works for chunked simulations!")
    _, assembly_metadata = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)

    i, n_assemblies = 0, np.zeros_like(stim_times, dtype=int)
    for seed, t_start, t_end in zip(project_metadata["seeds"], t_chunks[:-1], t_chunks[1:]):
        stim_times_chunk = stim_times[(t_start <= stim_times) & (stim_times < t_end)]
        t_bins = spike_matrices[seed].t_bins
        clusters = assembly_metadata["seed%i" % seed]["clusters"]
        assert len(t_bins) == len(clusters)
        split_clusters = np.split(clusters, np.searchsorted(t_bins, stim_times_chunk))[1:]
        for pattern_clusters in split_clusters:
            n_assemblies[i] = len(np.unique(pattern_clusters))
            i += 1

    fig_name = os.path.join(config.fig_path, "n_assemblies.png")
    plot_n_assemblies(stim_times, patterns, n_assemblies, t_chunks, fig_name)


if __name__ == "__main__":
    config_path = "../configs/v7_10mins.yaml"
    count_assemblies(config_path)
