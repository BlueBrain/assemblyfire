# -*- coding: utf-8 -*-
"""
Correlate consensus assembly membership and single cell features
(spike time reliability and mean+/-std of spike times within time bins)
last modified: András Ecker 11.2021
"""

import os
import numpy as np

from assemblyfire.config import Config
from assemblyfire.utils import load_single_cell_features_from_h5, load_consensus_assemblies_from_h5,\
                               get_sim_path, get_figure_asthetics
from assemblyfire.plots import plot_consensus_r_spike, plot_consensus_t_in_bin,\
                               plot_coreness_r_spike, plot_coreness_t_in_bin


def consensus_vs_single_cell_features(config_path):
    """Loads in consensus assemblies and single cell features from h5 file
    and plots the distributions of spike time reliability for consensus assemblies"""
    config = Config(config_path)

    # loading single cell features and consensus assemblies from h5
    single_cell_features, _ = load_single_cell_features_from_h5(config.h5f_name, prefix=config.h5_prefix_single_cell)
    consensus_assemblies = load_consensus_assemblies_from_h5(config.h5f_name,
                                                             prefix=config.h5_prefix_consensus_assemblies)
    all_gids = single_cell_features.gids
    consensus_gids = [assembly.gids for _, assembly in consensus_assemblies.items()]
    union_gids = [assembly.union.gids for _, assembly in consensus_assemblies.items()]

    # consensus assembly membership vs. spike time reliability
    r_spikes = single_cell_features.r_spikes
    r_spikes[r_spikes == 0] = np.nan
    consenus_r_spikes = [r_spikes[np.searchsorted(all_gids, gids)] for gids in consensus_gids]
    union_r_spikes = [r_spikes[np.searchsorted(all_gids, gids)] for gids in union_gids]
    fig_name = os.path.join(config.fig_path, "consensus_r_spikes.png")
    plot_consensus_r_spike(consenus_r_spikes, union_r_spikes, r_spikes, fig_name)
    # coreness vs. spike time reliability
    coreness = [assembly.coreness for _, assembly in consensus_assemblies.items()]
    union_r_spikes = [r_spikes[np.searchsorted(all_gids, gids)] for gids in union_gids]
    fig_name = os.path.join(config.fig_path, "coreness_r_spikes.png")
    plot_coreness_r_spike(union_r_spikes, coreness, fig_name)

    # consensus assembly membership vs. spike time in bin
    mean_ts = single_cell_features.mean_ts
    consenus_mean_ts = [mean_ts[np.searchsorted(all_gids, gids)] for gids in consensus_gids]
    std_ts = single_cell_features.std_ts
    consenus_std_ts = [std_ts[np.searchsorted(all_gids, gids)] for gids in consensus_gids]
    depths, ystuff = get_figure_asthetics(get_sim_path(config.root_path).iloc[0], config.target)
    fig_name = os.path.join(config.fig_path, "consensus_t_in_bin.png")
    plot_consensus_t_in_bin(consensus_gids, all_gids, consenus_mean_ts, consenus_std_ts,
                            mean_ts, std_ts, ystuff, depths, config.bin_size, fig_name)
    # coreness vs. spike time in bin
    union_mean_ts = [mean_ts[np.searchsorted(all_gids, gids)] for gids in union_gids]
    union_std_ts = [std_ts[np.searchsorted(all_gids, gids)] for gids in union_gids]
    fig_name = os.path.join(config.fig_path, "coreness_t_in_bin.png")
    plot_coreness_t_in_bin(union_mean_ts, union_std_ts, coreness, config.bin_size, fig_name)


if __name__ == "__main__":
    consensus_vs_single_cell_features("../configs/v7_bbp-workflow.yaml")
