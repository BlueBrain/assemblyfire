# -*- coding: utf-8 -*-
"""
Script to add consensus assembly botany stuff (e.g. mtype composition)
last modified: András Ecker 11.2020
"""

import os
import numpy as np

from assemblyfire.spikes import SpikeMatrixGroup
from assemblyfire.connectivity import ConnectivityMatrix
from assemblyfire.utils import map_gids_to_depth, get_layer_boundaries, load_consensus_assemblies_from_h5
from assemblyfire.plots import plot_consensus_mtypes, plot_consensus_in_degree


def _get_mtypes(spikes):
    """Gets excitatory gids and corresponding mtypes from BlueConfig
    :param spikes: SpikeMatrixGroup object that stores config params and simulation metadata"""
    from assemblyfire.spikes import get_bluepy_simulation
    from assemblyfire.utils import get_E_gids, get_mtypes
    sim = get_bluepy_simulation(spikes.get_blueconfig_path(spikes.seeds[0]))
    all_gids = get_E_gids(sim.circuit, sim.target)
    return all_gids, np.asarray(get_mtypes(sim.circuit, all_gids))


def get_indegrees(connectivity_array):
    """Gets indegrees for all neurons
    :param connectivity_array: connectivity matrix as np.array"""
    return np.sum(connectivity_array, axis=0)


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

    # TODO move this part to topology, once it has a proper structure
    connectivity_matrix = ConnectivityMatrix.from_h5(spikes.h5f_name, group_name="full_matrix", prefix="connectivity")
    consensus_in_degrees = [get_indegrees(connectivity_matrix.subarray(gids)) for gids in consensus_gids]
    blueconfig_path = spikes.get_blueconfig_path(spikes.seeds[0])
    control_in_degrees_depth = [get_indegrees(connectivity_matrix.sample_depth_profile(blueconfig_path, gids))
                                for gids in consensus_gids]
    control_in_degrees_mtypes = [get_indegrees(connectivity_matrix.sample_mtype_composition(blueconfig_path, gids))
                                 for gids in consensus_gids]
    fig_name = os.path.join(spikes.fig_path, "consensus_in_degrees.png")
    plot_consensus_in_degree(consensus_in_degrees, control_in_degrees_depth, control_in_degrees_mtypes, fig_name)


if __name__ == "__main__":
    consensus_botany("../configs/100p_depol_simmat.yaml")
