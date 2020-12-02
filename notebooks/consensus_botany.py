# -*- coding: utf-8 -*-
"""
Script to add consensus assembly botany stuff (e.g. mtype composition)
last modified: András Ecker 12.2020
"""

import os
import numpy as np

from assemblyfire.spikes import SpikeMatrixGroup
from assemblyfire.topology import NetworkAssembly
from assemblyfire.utils import map_gids_to_depth, get_layer_boundaries, load_consensus_assemblies_from_h5
from assemblyfire.plots import plot_consensus_mtypes, plot_consensus_in_degree


def consensus_botany(config_path):
    """Loads in consensus assemblies and plots unions' and cores' depth profile and mtype composition"""

    spikes = SpikeMatrixGroup(config_path)  # only used for the config part...
    consensus_assemblies = load_consensus_assemblies_from_h5(spikes.h5f_name, prefix="consensus", load_metadata=False)
    network = NetworkAssembly.from_h5(spikes.h5f_name, group_name="full_matrix", prefix="connectivity")

    all_gids = network.gids
    consensus_gids = [assembly.gids for _, assembly in consensus_assemblies.items()]
    union_gids = [assembly.union.gids for _, assembly in consensus_assemblies.items()]

    mtypes = network.mtypes  # this is a bit weird that it's stored in NetworkAssembly...
    consensus_mtypes = [mtypes[np.searchsorted(all_gids, gids)] for gids in consensus_gids]
    union_mtypes = [mtypes[np.searchsorted(all_gids, gids)] for gids in union_gids]

    depths = map_gids_to_depth(spikes.get_blueconfig_path(spikes.seeds[0]), all_gids)
    ystuff = get_layer_boundaries(spikes.get_blueconfig_path(spikes.seeds[0]))
    fig_name = os.path.join(spikes.fig_path, "consensus_mtypes.png")
    plot_consensus_mtypes(union_gids, union_mtypes, consensus_gids, all_gids, consensus_mtypes, mtypes,
                          ystuff, depths, fig_name)

    # TODO move this to `assembly_topolgy.py`
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
    fig_name = os.path.join(spikes.fig_path, "consensus_in_degrees.png")
    plot_consensus_in_degree(consensus_in_degrees, control_in_degrees_depth, control_in_degrees_mtypes, fig_name)
    fig_name = os.path.join(spikes.fig_path, "consensus_in_degrees_union.png")
    plot_consensus_in_degree(consensus_in_degrees, control_in_degrees_depth_union,
                             control_in_degrees_mtypes_union, fig_name)


if __name__ == "__main__":
    consensus_botany("../configs/100p_depol_simmat.yaml")
