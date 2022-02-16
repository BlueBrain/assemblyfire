# -*- coding: utf-8 -*-
"""
Consensus assembly botany stuff (so far: mtype composition and depth profile of core vs. union)
last modified: András Ecker 11.2021
"""

import os
import numpy as np

from assemblyfire.config import Config
from assemblyfire.topology import AssemblyTopology
from assemblyfire.utils import get_sim_path, get_figure_asthetics, load_consensus_assemblies_from_h5
from assemblyfire.plots import plot_consensus_mtypes


def consensus_botany(config_path):
    """Loads in consensus assemblies and plots unions' and cores' depth profile and mtype composition"""

    config = Config(config_path)
    consensus_assemblies = load_consensus_assemblies_from_h5(config.h5f_name,
                                                             prefix=config.h5_prefix_consensus_assemblies)
    conn_mat = AssemblyTopology.from_h5(config.h5f_name, prefix=config.h5_prefix_connectivity)

    all_gids = conn_mat.gids
    consensus_gids = [assembly.gids for _, assembly in consensus_assemblies.items()]
    union_gids = [assembly.union.gids for _, assembly in consensus_assemblies.items()]
    mtypes = conn_mat.mtype  # this is a bit weird that it's stored in NetworkAssembly...
    consensus_mtypes = [mtypes[np.searchsorted(all_gids, gids)] for gids in consensus_gids]
    union_mtypes = [mtypes[np.searchsorted(all_gids, gids)] for gids in union_gids]

    depths, ystuff = get_figure_asthetics(get_sim_path(config.root_path).iloc[0], config.target)
    fig_name = os.path.join(config.fig_path, "consensus_mtypes.png")
    plot_consensus_mtypes(union_gids, union_mtypes, consensus_gids, all_gids, consensus_mtypes, mtypes,
                          ystuff, depths, fig_name)


if __name__ == "__main__":
    consensus_botany("../configs/v7_10mins.yaml")
