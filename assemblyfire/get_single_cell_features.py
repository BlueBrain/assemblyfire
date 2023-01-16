"""
Main run function for finding getting single cell features
last modified: András Ecker 01.2023
"""

import os
import logging

from assemblyfire.spikes import SpikeMatrixGroup, single_cell_features_to_h5
from assemblyfire.utils import get_sim_path, get_neuron_locs
from assemblyfire.plots import plot_r_spikes

L = logging.getLogger("assemblyfire")


def run(config_path):
    """
    Loads in project related info from yaml config file, loads in simulations and calculates spike time reliability
    TODO: gather ideas and implement more single cell features (that are defined over repetitions aka. across seeds)
    :param config_path: str - path to project config file
    """

    spikes = SpikeMatrixGroup(config_path)
    L.info(" Load in spikes from %s" % spikes.root_path)

    L.info(" Single cell features will be saved to: %s" % spikes.h5f_name)
    gids, r_spikes = spikes.get_spike_time_reliability()
    single_cell_features_to_h5(spikes.h5f_name, gids, r_spikes, prefix=spikes.h5_prefix_single_cell)

    L.info(" Figures will be saved to: %s" % spikes.fig_path)
    nrn_loc_df = get_neuron_locs(get_sim_path(spikes.root_path).iloc[0], spikes.target)
    plot_r_spikes(gids, r_spikes, nrn_loc_df, os.path.join(spikes.fig_path, "r_spikes.png"))
