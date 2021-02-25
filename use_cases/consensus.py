# -*- coding: utf-8 -*-
"""
Exemplar script which calculates simplices in consensus assemblies for specific use cases
(At this point everything should be saved in the general assemblyfire format, so nothing fancy/extra here)
last modified: András Ecker 02.2021
"""

import os
from assemblyfire.utils import load_assemblies_from_h5, load_consensus_assemblies_from_h5
from assemblyfire.assemblies import consensus_over_seeds_hamming
from assemblyfire.topology import NetworkAssembly, simplex_counts_consensus
from assemblyfire.plots import plot_simplex_counts_consensus

fig_path = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/figures/toposample"


if __name__ == "__main__":

    h5f_name = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/assemblyfire/use_cases/assemblies.h5"

    # load assemblies from file, create consensus assemblies, saving them to h5, and loading again
    assembly_grp_dict = load_assemblies_from_h5(h5f_name, prefix="assemblies", load_metadata=False)
    consensus_over_seeds_hamming(assembly_grp_dict, h5f_name,
                                 h5_prefix="consensus", fig_path=fig_path)
    consensus_assembly_dict = load_consensus_assemblies_from_h5(h5f_name, prefix="consensus", load_metadata=False)

    network = NetworkAssembly.from_h5(h5f_name, group_name="full_matrix", prefix="connectivity")
    simplex_counts, simplex_counts_control = simplex_counts_consensus(consensus_assembly_dict, network, N=1)

    fig_name = os.path.join(fig_path, "simplex_counts_consensus.png")
    plot_simplex_counts_consensus(simplex_counts, simplex_counts_control, fig_name)


