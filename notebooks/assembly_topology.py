# -*- coding: utf-8 -*-
"""
(In degrees and) Simplex counts of assemblies and consensus assemblies
last modified: András Ecker 11.2021
"""

import os

from assemblyfire.config import Config
from assemblyfire.topology import NetworkAssembly, in_degree_assemblies, simplex_counts_assemblies,\
                                  simplex_counts_consensus
from assemblyfire.utils import load_assemblies_from_h5, load_consensus_assemblies_from_h5
from assemblyfire.plots import plot_simplex_counts_seed, plot_in_degrees, plot_simplex_counts_consensus


def assembly_topology(config_path):
    """Loads in assemblies and plots in degrees and simplex counts (seed by seed)"""

    config = Config(config_path)
    network = NetworkAssembly.from_h5(config.h5f_name, prefix=config.h5_prefix_connectivity)
    assembly_grp_dict, _ = load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)
    in_degrees, in_degrees_control = in_degree_assemblies(assembly_grp_dict, network)
    for seed, in_degree in in_degrees.items():
        fig_name = os.path.join(config.fig_path, "in_degrees_%s.png" % seed)
        plot_in_degrees(in_degree, in_degrees_control[seed], fig_name)

    simplex_counts, simplex_counts_control = simplex_counts_assemblies(assembly_grp_dict, network)
    for seed, simplices in simplex_counts.items():
        fig_name = os.path.join(config.fig_path, "simplex_counts_%s.png" % seed)
        plot_simplex_counts_seed(simplices, simplex_counts_control[seed], fig_name)

    consensus_assembly_dict = load_consensus_assemblies_from_h5(config.h5f_name,
                                                                prefix=config.h5_prefix_consensus_assemblies)
    simplex_counts, simplex_counts_control = simplex_counts_consensus(consensus_assembly_dict, network, n_ctrls=1)
    fig_name = os.path.join(config.fig_path, "simplex_counts_consensus.png")
    plot_simplex_counts_consensus(simplex_counts, simplex_counts_control, fig_name)

    '''TODO: adapt prev. implementation of consensus assembly in degree
    consensus_in_degrees = [network.degree(gids, kind="in") for gids in consensus_gids]
    control_in_degrees_depth = [network.degree(network.sample_gids_depth_profile(gids), kind="in")
                                for gids in consensus_gids]
    control_in_degrees_depth_union = []
    for i, gids in enumerate(consensus_gids):
        control_in_degrees_depth_union.append(network.degree(
            network.sample_gids_depth_profile(gids, sub_gids=union_gids[i]), kind="in"))
    control_in_degrees_mtypes = [network.degree(network.sample_gids_mtype_composition(gids), kind="in")
                                 for gids in consensus_gids]
    control_in_degrees_mtypes_union = []
    for i, gids in enumerate(consensus_gids):
        control_in_degrees_mtypes_union.append(network.degree(
            network.sample_gids_mtype_composition(gids, sub_gids=union_gids[i]), kind="in"))
    fig_name = os.path.join(config.fig_path, "consensus_in_degrees.png")
    plot_consensus_in_degree(consensus_in_degrees, control_in_degrees_depth, control_in_degrees_mtypes, fig_name)
    fig_name = os.path.join(config.fig_path, "consensus_in_degrees_union.png")
    plot_consensus_in_degree(consensus_in_degrees, control_in_degrees_depth_union,
                             control_in_degrees_mtypes_union, fig_name)
    '''


if __name__ == "__main__":
    assembly_topology("../configs/v7_bbp-workflow.yaml")
