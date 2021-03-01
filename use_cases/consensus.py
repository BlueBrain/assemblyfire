# -*- coding: utf-8 -*-
"""
Exemplar script which calculates simplices in consensus assemblies for this specific use cases
(At this point everything should be saved in the general assemblyfire format, so nothing fancy/extra here)
last modified: András Ecker 02.2021
"""

import os
import numpy as np
from assemblyfire.utils import load_assemblies_from_h5, load_consensus_assemblies_from_h5
from assemblyfire.assemblies import consensus_over_seeds_hamming
from assemblyfire.topology import NetworkAssembly, simplex_counts_consensus
from assemblyfire.plots import plot_simplex_counts_consensus, plot_cons_cluster_seqs
from find_assemblies import load_patterns

data_dir = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/assemblyfire/use_cases/input_data"
fig_path = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/figures/toposample"


def _str2int(s):
    """Returns integer part of string (only to be used in this setup, not a general function)"""
    return int(''.join(i for i in s if i.isdigit()))


def _find_assembly(consensus_assembly_dict, seed, assembly_id):
    """Helper function to find assembly in consensus assemblies (returns -1 if not found)"""
    for cons_id, cons_assembly in consensus_assembly_dict.items():
        for assembly in cons_assembly.instantiations:
            if assembly.idx[1] == seed:
                if assembly.idx[0] == assembly_id:
                    return _str2int(cons_id)
    return -1


def consensus_assembly_cluster_seq(consensus_assembly_dict, assembly_metadata, data_dir):
    """Recolor cluster seqs. based on consensus assemblies"""

    # the beginning is just duplicated from `use_cases/find_assemblies.py` ...
    stim_times, patterns = load_patterns(data_dir)
    t_full = stim_times[-1] + 200
    t_slices = np.arange(0, t_full+30000, 30000)
    for i, (t_start, t_end) in enumerate(zip(t_slices[:-1], t_slices[1:])):
        print(i)
        t_bins = np.arange(t_start, t_end + 20., 20.)  # bin size = 20 ms from the rest of the use case
        idx = np.where((t_start <= stim_times) & (stim_times < t_end))[0]

        # from here it's not that hacky as it's just accessing saved files...
        orig_clusters = assembly_metadata["seed%i" % i]["clusters"]
        cons_clusters = np.empty(orig_clusters.shape, dtype=np.int)
        for cluster in np.unique(orig_clusters):
            cons_clusters[orig_clusters == cluster] = _find_assembly(consensus_assembly_dict,
                                                                     seed=i, assembly_id=cluster)
        if i == 29:
            t_bins = t_bins[:len(cons_clusters)]  # the last slice is shorter
        if cons_clusters.shape[0] == 1499:
            cons_clusters = np.insert(cons_clusters, 1499, -1)  # I have no clue why this happens...
        fig_name = os.path.join(fig_path, "cons_cluster_seq_seed%i.png" % i)
        plot_cons_cluster_seqs(cons_clusters, t_bins, stim_times[idx], patterns[idx],
                               n_clusters=len(consensus_assembly_dict.keys()), fig_name=fig_name)


if __name__ == "__main__":

    h5f_name = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/assemblyfire/use_cases/assemblies.h5"

    # load assemblies from file, create consensus assemblies, saving them to h5, and loading again
    assembly_grp_dict, assembly_metadata = load_assemblies_from_h5(h5f_name, prefix="assemblies", load_metadata=True)
    consensus_over_seeds_hamming(assembly_grp_dict, h5f_name,
                                 h5_prefix="consensus", fig_path=fig_path)
    consensus_assembly_dict = load_consensus_assemblies_from_h5(h5f_name, prefix="consensus")

    consensus_assembly_cluster_seq(consensus_assembly_dict, assembly_metadata, data_dir)

    network = NetworkAssembly.from_h5(h5f_name, group_name="full_matrix", prefix="connectivity")
    simplex_counts, simplex_counts_control = simplex_counts_consensus(consensus_assembly_dict, network, N=1)
    fig_name = os.path.join(fig_path, "simplex_counts_consensus.png")
    plot_simplex_counts_consensus(simplex_counts, simplex_counts_control, fig_name)


