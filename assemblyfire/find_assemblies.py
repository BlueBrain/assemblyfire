"""
Main run function for finding cell assemblies in spiking data
last modified: András Ecker 10.2022
"""

import logging

from assemblyfire.utils import ensure_dir, get_sim_path, get_neuron_locs
from assemblyfire.spikes import SpikeMatrixGroup
from assemblyfire.clustering import cluster_spikes, detect_assemblies

L = logging.getLogger("assemblyfire")


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

    L.info(" Preprocessed spikes and assemblies will be saved to: %s" % spikes.h5f_name)
    spike_matrix_dict, project_metadata = spikes.get_sign_spike_matrices()

    L.info(" Cluster time bins via hierarchical clustering...")
    clusters_dict = cluster_spikes(spike_matrix_dict, spikes.overwrite_seeds, project_metadata, spikes.fig_path)

    L.info(" Detecting assemblies within clustered time bins and saving them to file...")
    nrn_loc_df = get_neuron_locs(get_sim_path(spikes.root_path).iloc[0], spikes.target)
    detect_assemblies(spike_matrix_dict, clusters_dict, spikes.core_cell_th_pct, spikes.h5f_name,
                      spikes.h5_prefix_assemblies, nrn_loc_df, spikes.fig_path)
