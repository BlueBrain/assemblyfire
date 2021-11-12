# -*- coding: utf-8 -*-
"""
Main run function for finding getting single cell features
(spike time reliability and mean+/-std of spike times within time bins)
last modified: András Ecker 11.2020
"""

import os
import logging

from assemblyfire.spikes import SpikeMatrixGroup, single_cell_features_to_h5
from assemblyfire.utils import get_figure_asthetics
from assemblyfire.plots import plot_single_cell_features

L = logging.getLogger("assemblyfire")


def run(config_path):
    """
    Loads in project related info from yaml config file, loads in simulations and calculates
    spike time reliability and mean+/-std of spike times within time bins across seeds
    :param config_path: str - path to project config file
    """

    spikes = SpikeMatrixGroup(config_path)
    L.info(" Load in spikes from %s" % spikes.root_path)

    L.info(" Single cell features will be saved to: %s" % spikes.h5f_name)
    gids_spikes, mean_ts, std_ts = spikes.get_mean_std_ts_in_bin()
    gids_r, r_spikes = spikes.get_spike_time_reliability()
    assert (gids_spikes == gids_r).all()
    single_cell_features_to_h5(spikes.h5f_name, gids_r, r_spikes, mean_ts, std_ts,
                               prefix=spikes.h5_prefix_spikes)

    L.info(" Figures will be saved to: %s" % spikes.fig_path)
    depths, ystuff = get_figure_asthetics(spikes.load_sim_path().iloc[0], spikes.target)
    fig_name = os.path.join(spikes.fig_path, "single_cell_features.png")
    plot_single_cell_features(gids_r, r_spikes, mean_ts, std_ts, ystuff, depths, fig_name)
