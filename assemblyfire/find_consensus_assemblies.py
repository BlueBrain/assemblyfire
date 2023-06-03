"""
Main run function for finding consensus assemblies
last modified: András Ecker 01.2023
"""

import logging

from assemblyfire.utils import load_assemblies_from_h5, get_nrn_df
from assemblyfire.spikes import SpikeMatrixGroup
from assemblyfire.clustering import cluster_spikes, detect_assemblies
from assemblyfire.assemblies import consensus_over_seeds

L = logging.getLogger("assemblyfire")


def run(config_path, average):
    """
    Loads in project related info from yaml config file, finds consensus assemblies and saves them to h5
    :param config_path: str - path to project config file
    """
    spikes = SpikeMatrixGroup(config_path)
    L.info(" Load in assemblies from %s" % spikes.h5f_name)
    assembly_grp_dict, _ = load_assemblies_from_h5(spikes.h5f_name, spikes.h5_prefix_assemblies)
    L.info(" Creating consensus assemblies and saving them to the same file")
    consensus_over_seeds(assembly_grp_dict, spikes.h5f_name, spikes.h5_prefix_consensus_assemblies, spikes.fig_path)

    if average:
        L.info(" Load in spikes from %s, average them and detect average assemblies " % spikes.root_path)
        spike_matrix_dict, project_metadata = spikes.get_mean_sign_spike_matrix()
        clusters_dict = cluster_spikes(spike_matrix_dict, spikes.overwrite_seeds, project_metadata, spikes.fig_path)
        nrn_loc_df = get_nrn_df(spikes.h5f_name, spikes.h5_prefix_connectivity, spikes.root_path, spikes.target)
        detect_assemblies(spike_matrix_dict, clusters_dict, spikes.core_cell_th_pct, spikes.h5f_name,
                          spikes.h5_prefix_avg_assemblies, nrn_loc_df, spikes.fig_path)

