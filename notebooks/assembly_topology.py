# -*- coding: utf-8 -*-
"""
In degrees and simplex counts of assemblies and TODO: consensus assemblies
last modified: András Ecker 12.2020
"""

import os

from assemblyfire.spikes import SpikeMatrixGroup
from assemblyfire.topology import NetworkAssembly, in_degree_assemblies, simplex_counts_assemblies
from assemblyfire.utils import load_assemblies_from_h5
from assemblyfire.plots import plot_simplex_counts_seed, plot_in_degrees


def assembly_topology(config_path):
    """Loads in assemblies and plots in degrees and simplex counts (seed by seed)"""

    spikes = SpikeMatrixGroup(config_path)  # only used for the config part...
    assembly_grp_dict = load_assemblies_from_h5(spikes.h5f_name, spikes.h5_prefix_assemblies, load_metadata=False)
    network = NetworkAssembly.from_h5(spikes.h5f_name, group_name="full_matrix",
                                      prefix=spikes.config["h5_out"]["prefixes"]["connectivity"])

    in_degrees, in_degrees_control = in_degree_assemblies(assembly_grp_dict, network)
    for seed, in_degree in in_degrees.items():
        fig_name = os.path.join(spikes.fig_path, "in_degrees_%s.png" % seed)
        plot_in_degrees(in_degree, in_degrees_control[seed], fig_name)

    simplex_counts, simplex_counts_control = simplex_counts_assemblies(assembly_grp_dict, network)
    for seed, simplices in simplex_counts.items():
        fig_name = os.path.join(spikes.fig_path, "simplex_counts_%s.png" % seed)
        plot_simplex_counts_seed(simplices, simplex_counts_control[seed], fig_name)


if __name__ == "__main__":
    assembly_topology("../configs/100p_depol_simmat.yaml")
