# -*- coding: utf-8 -*-
"""
Main run function for getting (and saving) connectivity matrix
last modified: András Ecker 11.2020
"""

import logging

from assemblyfire.spikes import SpikeMatrixGroup
from assemblyfire.connectivity import ConnectivityMatrix

L = logging.getLogger("assemblyfire")


def run(config_path):
    """
    Loads in project related info from yaml config file, gets connectivity matrix and saves it to h5 file
    :param config_path: str - path to project config file
    """

    # only used for the config part...
    spikes = SpikeMatrixGroup(config_path)
    h5f_name = spikes.h5f_name
    prefix = spikes.config["h5_out"]["prefixes"]["connectivity"]

    L.info("Loading in connectivity matrix and saving it to: %s" % h5f_name)
    connectivity_matrix = ConnectivityMatrix.from_bluepy(spikes.get_blueconfig_path(spikes.seeds[0]))
    connectivity_matrix.to_h5(h5f_name, group_name="full_matrix", prefix=prefix)

