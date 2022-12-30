"""
Visualizes the mapping of assemblies (across seeds) to their consensus assembly
last modified: András Ecker 12.2022
"""

import os
import numpy as np

from assemblyfire.config import Config
from assemblyfire.utils import load_assemblies_from_h5, load_consensus_assemblies_from_h5, read_cluster_seq_data
from assemblyfire.plots import plot_cons_cluster_seqs, plot_cons_cluster_seqs_all_seeds


def _create_cons_inst_lookup(consensus_assemblies):
    """Creates lookup table from consensus assembly instantiations (for easier mapping later)"""
    return {i: [assembly.idx for assembly in consensus_assemblies["cluster%i" % i].instantiations]
            for i in range(len(consensus_assemblies))}


def _find_cons_cluster_id(assembly_id, consensus_instantiations):
    """Finds assembly id in consensus instantiations"""
    for consensus_cluster_id, instantiations in consensus_instantiations.items():
        if assembly_id in instantiations:
            return consensus_cluster_id


def consensus_cluster_seqs(config_path):
    """Re-colors cluster (of time bin) sequences based on the IDs of the consensus assemblies"""

    config = Config(config_path)
    assembly_grp_dict, _ = load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)
    consensus_assemblies = load_consensus_assemblies_from_h5(config.h5f_name, config.h5_prefix_consensus_assemblies)
    consensus_instantiations = _create_cons_inst_lookup(consensus_assemblies)
    n_clusters = len(consensus_instantiations)
    metadata = read_cluster_seq_data(config.h5f_name)
    stim_times, patterns = metadata["stim_times"], metadata["patterns"]

    all_clusters = {}
    for seed, asassembly_grp in assembly_grp_dict.items():
        orig_clusters, t_bins = metadata["clusters"][seed], metadata["t_bins"][seed]
        clusters = -1 * np.ones_like(orig_clusters)
        for assembly in asassembly_grp.assemblies:
            clusters[orig_clusters == assembly.idx[0]] = _find_cons_cluster_id(assembly.idx, consensus_instantiations)
        all_clusters[seed] = clusters

        fig_name = os.path.join(config.fig_path, "consensus_cluster_seq_%s.png" % seed)
        plot_cons_cluster_seqs(clusters, t_bins, stim_times, patterns, n_clusters, fig_name)
    fig_name = os.path.join(config.fig_path, "consensus_cluster_seqs_all_seeds.png")
    plot_cons_cluster_seqs_all_seeds(all_clusters, metadata["t_bins"], stim_times, patterns, n_clusters, fig_name)


if __name__ == "__main__":
    consensus_cluster_seqs("../configs/v7_10seeds_np.yaml")
