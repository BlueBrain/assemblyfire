# -*- coding: utf-8 -*-
"""
Script to add consensus assembly botany stuff (e.g. mtype composition)
last modified: András Ecker 11.2020
"""

import os
import numpy as np

from assemblyfire.spikes import SpikeMatrixGroup
from assemblyfire.utils import map_gids_to_depth, get_layer_boundaries, load_consensus_assemblies_from_h5
from assemblyfire.plots import plot_consensus_mtypes


def _get_mtypes(spikes):
    """Gets excitatory gids and corresponding mtypes from BlueConfig
    :param spikes: SpikeMatrixGroup object that stores config params and simulation metadata"""
    from assemblyfire.spikes import get_bluepy_simulation
    from assemblyfire.utils import get_E_gids, get_mtypes
    sim = get_bluepy_simulation(spikes.get_blueconfig_path(spikes.seeds[0]))
    all_gids = get_E_gids(sim.circuit, sim.target)
    return all_gids, np.asarray(get_mtypes(sim.circuit, all_gids))

def consensus_botany(config_path):
    """Loads in consensus assemblies and plots unions' and cores' depth profile and mtype composition"""

    # only used for the config part...
    spikes = SpikeMatrixGroup(config_path)

    # loading consensus assemblies from h5
    consensus_assemblies, _ = load_consensus_assemblies_from_h5(spikes.h5f_name, prefix="consensus")
    consensus_gids = [assembly.gids for _, assembly in consensus_assemblies.items()]
    union_gids = [assembly.union.gids for _, assembly in consensus_assemblies.items()]
    # loading all gids from the circuit (just for comparison baseline)
    all_gids, mtypes = _get_mtypes(spikes)
    consensus_mtypes = [mtypes[np.searchsorted(all_gids, gids)] for gids in consensus_gids]
    union_mtypes = [mtypes[np.searchsorted(all_gids, gids)] for gids in union_gids]

    depths = map_gids_to_depth(spikes.get_blueconfig_path(spikes.seeds[0]), all_gids)
    ystuff = get_layer_boundaries(spikes.get_blueconfig_path(spikes.seeds[0]))
    fig_name = os.path.join(spikes.fig_path, "consensus_mtypes.png")
    plot_consensus_mtypes(union_gids, union_mtypes, consensus_gids, all_gids, consensus_mtypes, mtypes,
                          ystuff, depths, fig_name)


if __name__ == "__main__":
    consensus_botany("../configs/100p_depol_simmat.yaml")
