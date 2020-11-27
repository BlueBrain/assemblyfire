# -*- coding: utf-8 -*-
"""
Correlate consensus assembly membership and single cell features
(spike time reliability and mean+/-std of spike times within time bins)
"""

import os
import numpy as np

from assemblyfire.spikes import SpikeMatrixGroup
from assemblyfire.utils import map_gids_to_depth, get_layer_boundaries,\
                               load_single_cell_features_from_h5, load_consensus_assemblies_from_h5
from assemblyfire.plots import plot_consensus_r_spike, plot_consensus_t_in_bin


def consensus_vs_single_cell_features(config_path):
    """Loads in consensus assemblies and single cell features from h5 file
    and plots the distributions of spike time reliability for consensus assemblies"""

    # only used for the config part...
    spikes = SpikeMatrixGroup(config_path)

    # loading single cell features and consensus assemblies from h5
    single_cell_features, _ = load_single_cell_features_from_h5(spikes.h5f_name, prefix="spikes")
    consensus_assemblies, _ = load_consensus_assemblies_from_h5(spikes.h5f_name, prefix="consensus")
    all_gids = single_cell_features.gids
    consensus_gids = [assembly.gids for _, assembly in consensus_assemblies.items()]

    """
    # consensus assembly membership vs. spike time reliability
    r_spikes = single_cell_features.r_spikes
    r_spikes[r_spikes == 0] = np.nan
    consenus_r_spikes = [r_spikes[np.searchsorted(all_gids, gids)] for gids in consensus_gids]
    fig_name = os.path.join(spikes.fig_path, "consensus_r_spikes.png")
    plot_consensus_r_spike(consenus_r_spikes, r_spikes, fig_name)
    """

    # consensus assembly membership vs. spike time in bin
    mean_ts = single_cell_features.mean_ts
    consenus_mean_ts = [mean_ts[np.searchsorted(all_gids, gids)] for gids in consensus_gids]
    std_ts = single_cell_features.std_ts
    consenus_std_ts = [std_ts[np.searchsorted(all_gids, gids)] for gids in consensus_gids]
    depths = map_gids_to_depth(spikes.get_blueconfig_path(spikes.seeds[0]))
    ystuff = get_layer_boundaries(spikes.get_blueconfig_path(spikes.seeds[0]))
    fig_name = os.path.join(spikes.fig_path, "consensus_t_in_bin.png")
    plot_consensus_t_in_bin(consensus_gids, all_gids, consenus_mean_ts, consenus_std_ts,
                            mean_ts, std_ts, ystuff, depths, fig_name)


if __name__ == "__main__":
    consensus_vs_single_cell_features("../configs/100p_depol_simmat.yaml")
