"""
Main run function for finding synapse cluster on assembly neurons
last modified: András Ecker 01.2022
"""

import os
import logging
from tqdm import tqdm
import numpy as np

from assemblyfire.config import Config
import assemblyfire.utils as utils
from assemblyfire.assemblies import AssemblyGroup
from assemblyfire.topology import AssemblyTopology
from assemblyfire.clustering import cluster_synapses
from assemblyfire.plots import plot_cond_rhos

L = logging.getLogger("assemblyfire")


def _get_assembly_mtypes(c, assembly, mtypes):
    """Helper function to select L5 TTPCs from assemblies"""
    mtypes = utils.get_mtypes(c, assembly.gids)
    return mtypes.loc[mtypes.isin(mtypes)].index.to_numpy()


def _get_rate_sorted_gids(sim, gids, t_start, t_end):
    """Sorts `gids` by firing rate"""
    _, spiking_gids = utils.get_spikes(sim, gids, t_start, t_end)
    gids, spike_counts = np.unique(spiking_gids, return_counts=True)
    sort_idx = np.argsort(spike_counts)[::-1]
    return gids[sort_idx]


def run(config_path, debug):
    """
    Loads in asssemblies from saved h5 file, and for each assembly samples L5_TTPCs
    (by in-degree for all assemblies and on top by firing rate for late assemblies),
    looks for synapse clusters on them and saves them to pickle files
    :param config_path: str - path to project config file
    :param debug: bool - to save figures for visual inspection
    """

    config = Config(config_path)
    L.info(" Load in assemblies and connectivity matrix from %s" % config.h5f_name)
    target_range, min_nsyns = config.syn_clustering_target_range, config.syn_clustering_min_nsyns
    mtypes, n_samples = config.syn_clustering_mtypes, config.syn_clustering_n_neurons_sample
    conn_mat = AssemblyTopology.from_h5(config.h5f_name,
                                        prefix=config.h5_prefix_connectivity, group_name="full_matrix")
    sim_paths = utils.get_sim_path(config.root_path)
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)

    L.info(" Detecting synapse clusters and saving them to files")
    for seed, assembly_grp in tqdm(assembly_grp_dict.items(), desc="Iterating over seeds"):
        cluster_dfs = {}
        sim = utils.get_bluepy_simulation(sim_paths.loc[int(seed.split("seed")[1])])
        for assembly in tqdm(assembly_grp.assemblies, desc="%s syn. clusters" % seed, leave=False):
            fig_dir = os.path.join(config.fig_path, "%s_debug" % seed) if debug else None
            target_mtype_gids = _get_assembly_mtypes(sim.circuit, assembly, mtypes)
            sorted_gids = assembly.gids[np.argsort(conn_mat.degree(assembly, kind="in"))[::-1]]
            post_gids = sorted_gids[np.in1d(sorted_gids, target_mtype_gids, assume_unique=True)][:n_samples]
            single_assembly_grp = AssemblyGroup([assembly], all_gids=assembly.gids)  # fake assembly "group"
            # get clusters and save them to pickle
            cluster_df = cluster_synapses(sim, post_gids, single_assembly_grp, assembly.idx[0],
                                          target_range, min_nsyns, fig_dir=fig_dir)
            utils.save_syn_clusters(config.root_path, assembly.idx, cluster_df)
            if assembly.idx[0] == 0:  # late assembly TODO: not hard code this distinction
                post_gids = _get_rate_sorted_gids(sim, target_mtype_gids, config.t_start, config.t_end)[:n_samples]
                late_cluster_df = cluster_synapses(sim, post_gids, assembly_grp, assembly.idx[0],
                                                   target_range, min_nsyns, fig_dir=fig_dir)
                utils.save_syn_clusters(config.root_path, assembly.idx, late_cluster_df, late_assembly=True)
            # some extra plotting
            cluster_df["rho"] = utils.get_syn_properties(sim.circuit, cluster_df.index.to_numpy(),
                                                         ["rho0_GB"])["rho0_GB"]
            cluster_dfs[assembly.idx[0]] = cluster_df
        fig_name = os.path.join(config.fig_path, "rho0_syn_clusts_%s.png" % seed)
        plot_cond_rhos(cluster_dfs, fig_name)

