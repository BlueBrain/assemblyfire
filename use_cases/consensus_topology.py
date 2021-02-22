# -*- coding: utf-8 -*-
"""
Exemplar script which calculates simplices in consensus assemblies for specific use cases
(At this point everything should be saved in the general assemblyfire format, so nothing fancy/extra here)
last modified: András Ecker 02.2021
"""

import os
from assemblyfire.utils import load_consensus_assemblies_from_h5
from assemblyfire.topology import NetworkAssembly, simplex_counts_consensus
from assemblyfire.plots import plot_simplex_counts_consensus

fig_path = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/figures/toposample"


if __name__ == "__main__":

    h5f_name = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/assemblyfire/use_cases/assemblies.h5"
    network = NetworkAssembly.from_h5(h5f_name, group_name="full_matrix", prefix="connectivity")
    consensus_assembly_dict = load_consensus_assemblies_from_h5(h5f_name, prefix="consensus", load_metadata=False)

    simplex_counts, simplex_counts_control = simplex_counts_consensus(consensus_assembly_dict, network, N=1)
    fig_name = os.path.join(fig_path, "simplex_counts_consensus.png")
    plot_simplex_counts_consensus(simplex_counts, simplex_counts_control, fig_name)


