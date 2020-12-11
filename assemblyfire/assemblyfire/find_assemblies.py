# -*- coding: utf-8 -*-
"""
Main run function for finding cell assemblies in spiking data
"""

import logging
from collections import namedtuple

from assemblyfire.utils import ensure_dir, map_gids_to_depth, get_layer_boundaries
from assemblyfire.spikes import SpikeMatrixGroup
from assemblyfire.clustering import cluster_spikes, detect_assemblies

L = logging.getLogger("assemblyfire")
FigureArgs = namedtuple("FigureArgs", ["stim_times", "patterns", "depths", "ystuff", "fig_path"])


def run(config_path):
    """
    Loads in project related info from yaml config file, bins raster and finds significant time bins (`spikes.py`)
    clusters time bins, detects and saves cell assemblies (`clustering.py`)
    :param config_path: str - path to project config file
    """

    spikes = SpikeMatrixGroup(config_path)
    L.info(" Load in spikes from %s" % spikes.root_path)

    L.info(" Figures will be saved to: %s" % spikes.fig_path)
    ensure_dir(spikes.fig_path)
    depths = map_gids_to_depth(spikes.get_blueconfig_path(spikes.seeds[0]))
    ystuff = get_layer_boundaries(spikes.get_blueconfig_path(spikes.seeds[0]))

    L.info(" Preprocessed spikes and assemblies will be saved to: %s" % spikes.h5f_name)
    spike_matrix_dict = spikes.get_sign_spike_matrices()
    L.info(" Cluster time bins via %s clustering..." % spikes.clustering_method)
    clusters_dict = cluster_spikes(spike_matrix_dict, method=spikes.clustering_method,
                                   FigureArgs=FigureArgs(spikes.stim_times, spikes.patterns,
                                                         depths, None, spikes.fig_path))
    L.info(" Detecting assemblies within clustered time bins and saving them to file...")
    detect_assemblies(spike_matrix_dict, clusters_dict, spikes.h5f_name, spikes.h5_prefix_assemblies,
                      FigureArgs=FigureArgs(None, None, depths, ystuff, spikes.fig_path))
