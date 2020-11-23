# -*- coding: utf-8 -*-
"""
Main run function for finding cell assemblies in spiking data
"""

import os
import logging
from collections import namedtuple

from assemblyfire.utils import ensure_dir, get_out_fname, map_gids_to_depth, get_layer_boundaries
from assemblyfire.spikes import SpikeMatrixGroup
from assemblyfire.clustering import cluster_spikes, detect_assemblies
from assemblyfire.assemblies import AssemblyProjectMetadata
from assemblyfire.plots import plot_rate

L = logging.getLogger("assemblyfire")
FigureArgs = namedtuple("FigureArgs", ["patterns", "depths", "ystuff", "fig_dir"])

def run(config_path):
    """
    Loads in project related info from yaml config file, bins raster and finds significant time bins (`spikes.py`)
    clusters time bins, detects and saves cell assemblies (`clustering.py`)
    :param config_path: str - path to project config file
    """

    L.info("Load in spikes and find significant time bins")
    spikes = SpikeMatrixGroup(config_path)
    spike_matrix_dict, rate_dict = spikes.get_spike_matrices()

    fig_dir = os.path.join(spikes.root_fig_path, config_path.split('/')[-1][:-5])
    ensure_dir(fig_dir)
    L.info("Figures will be saved to: %s" % fig_dir)
    # other plots are in `clustering.py/cluster_spikes() and detect_assemblies()`
    for seed, ThresholdedRate in rate_dict.items():
        fig_name = os.path.join(fig_dir, "rate_seed%i.png" % seed)
        plot_rate(ThresholdedRate.rate, ThresholdedRate.rate_th, spikes.t_start, spikes.t_end, fig_name)
    depths = map_gids_to_depth(spikes.get_blueconfig_path(spikes.seeds[0]))
    ystuff = get_layer_boundaries(spikes.get_blueconfig_path(spikes.seeds[0]))

    L.info("Cluster time bins via %s clustering" % spikes.clustering_method)
    clusters_dict = cluster_spikes(spike_matrix_dict, method=spikes.clustering_method,
                                   FigureArgs=FigureArgs(spikes.patterns, depths, None, fig_dir))

    h5f_name = get_out_fname(spikes.root_path, spikes.clustering_method)
    if os.path.isfile(h5f_name):
        os.remove(h5f_name)
    metadata = {"root_path": spikes.root_path, "seeds": spikes.seeds, "patterns": spikes.patterns}
    AssemblyProjectMetadata.to_h5(metadata, h5f_name, prefix="assemblies")
    L.info("Detecting assemblies within clustered time bins and saving them to file\n%s" % h5f_name)
    detect_assemblies(spike_matrix_dict, clusters_dict, h5f_name,
                      FigureArgs=FigureArgs(None, depths, ystuff, fig_dir))
