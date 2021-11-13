# -*- coding: utf-8 -*-
"""
Main run function for finding cell assemblies in spiking data
last modified: András Ecker 11.2021
"""

import logging
from collections import namedtuple

from assemblyfire.utils import ensure_dir, get_sim_path, get_figure_asthetics
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
    depths, ystuff = get_figure_asthetics(get_sim_path(spikes.root_path).iloc[0], spikes.target)

    L.info(" Preprocessed spikes and assemblies will be saved to: %s" % spikes.h5f_name)
    spike_matrix_dict, project_metadata = spikes.get_sign_spike_matrices()
    L.info(" Cluster time bins via %s clustering..." % spikes.spike_clustering_method)
    clusters_dict = cluster_spikes(spike_matrix_dict, method=spikes.spike_clustering_method,
                                   FigureArgs=FigureArgs(project_metadata["stim_times"], project_metadata["patterns"],
                                                         depths, None, spikes.fig_path))
    L.info(" Detecting assemblies within clustered time bins and saving them to file...")
    detect_assemblies(spike_matrix_dict, clusters_dict, spikes.h5f_name, spikes.h5_prefix_assemblies,
                      FigureArgs=FigureArgs(None, None, depths, ystuff, spikes.fig_path))
