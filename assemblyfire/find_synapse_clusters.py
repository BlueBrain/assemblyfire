"""
Main run function for finding synapse cluster on assembly neurons
last modified: András Ecker 09.2022
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


def _get_degree_sorted_assembly_gids_within_target(c, target_gids, conn_mat, assembly, mtype_list, n_samples):
    """Helper function to select in-degree sorted postsynaptic gids from assemblies (and target)"""
    sorted_assembly_gids = assembly.gids[np.argsort(conn_mat.degree(assembly, kind="in"))[::-1]]
    sorted_target_assembly_gids = sorted_assembly_gids[np.in1d(sorted_assembly_gids, target_gids, assume_unique=True)]
    mtypes = utils.get_mtypes(c, sorted_target_assembly_gids)
    return mtypes.loc[mtypes.isin(mtype_list)].index.to_numpy()[:n_samples]


def _get_rate_sorted_assembly_gids_within_target(sim, target_gids, assembly_gids, mtype_list, n_samples, t_start, t_end):
    """Helper function to select firing rate sorted postsynaptic gids from assemblies (and target)"""
    target_assembly_gids = assembly_gids[np.in1d(assembly_gids, target_gids, assume_unique=True)]
    _, spiking_gids = utils.get_spikes(sim, target_assembly_gids, t_start, t_end)
    gids, spike_counts = np.unique(spiking_gids, return_counts=True)
    sorted_target_assembly_gids = gids[np.argsort(spike_counts)[::-1]]
    mtypes = utils.get_mtypes(sim.circuit, sorted_target_assembly_gids)
    return mtypes.loc[mtypes.isin(mtype_list)].index.to_numpy()[:n_samples]


def run(config_path, debug):
    """
    Loads in asssemblies from saved h5 file, and for each assembly samples gids from a given mtype
    (by in-degree for all assemblies and on top by firing rate for late assemblies),
    looks for synapse clusters on them and saves them to pickle files
    :param config_path: str - path to project config file
    :param debug: bool - to save figures for visual inspection
    """

    config = Config(config_path)
    L.info(" Load in assemblies and connectivity matrix from %s" % config.h5f_name)
    sim_paths = utils.get_sim_path(config.root_path)
    c = utils.get_bluepy_cicruit(sim_paths.iloc[0])
    loc_df = utils.get_loc_df(config.syn_clustering_lookup_df_pklfname, c, config.target, config.syn_clustering_target)
    target_gids = loc_df["post_gid"].unique()
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)
    conn_mat = AssemblyTopology.from_h5(config.h5f_name,
                                        prefix=config.h5_prefix_connectivity, group_name="full_matrix")
    target_range, min_nsyns = config.syn_clustering_target_range, config.syn_clustering_min_nsyns
    mtypes, n_samples = config.syn_clustering_mtypes, config.syn_clustering_n_neurons_sample

    L.info(" Detecting synapse clusters and saving them to files")
    for seed, assembly_grp in tqdm(assembly_grp_dict.items(), desc="Iterating over seeds"):
        cluster_dfs = {}
        for assembly in tqdm(assembly_grp.assemblies, desc="%s syn. clusters" % seed, leave=False):
            gids = _get_degree_sorted_assembly_gids_within_target(c, target_gids, conn_mat,
                                                                  assembly, mtypes, n_samples)
            loc_df_gids = loc_df.loc[loc_df["post_gid"].isin(gids)]
            # create a fake assembly "group" in order to look for *within* assembly clusters only
            single_assembly_grp = AssemblyGroup([assembly], all_gids=assembly.gids)
            # get clusters and save them to pickle
            if debug:
                fig_dir = os.path.join(config.fig_path, "%s_debug" % seed)
                utils.ensure_dir(fig_dir)
                cluster_df = cluster_synapses(loc_df_gids, single_assembly_grp, target_range, min_nsyns,
                                              fig_dir=fig_dir, base_assembly_idx=assembly.idx[0], c=c)
            else:
                cluster_df = cluster_synapses(loc_df_gids, single_assembly_grp, target_range, min_nsyns)
            utils.save_syn_clusters(config.syn_clustering_save_dir, assembly.idx, cluster_df)

            sim = utils.get_bluepy_simulation(sim_paths.loc[int(seed.split("seed")[1])])
            if assembly.idx[0] == 0:  # late assembly TODO: not hard code this distinction
                gids = _get_rate_sorted_assembly_gids_within_target(sim, target_gids, assembly.gids, mtypes,
                                                                    n_samples, config.t_start, config.t_end)
                loc_df_gids = loc_df.loc[loc_df["post_gid"].isin(gids)]
                if debug:
                    late_cluster_df = cluster_synapses(loc_df_gids, assembly_grp, target_range, min_nsyns,
                                                       fig_dir=fig_dir, base_assembly_idx=assembly.idx[0], c=c)
                else:
                    late_cluster_df = cluster_synapses(loc_df_gids, assembly_grp, target_range, min_nsyns)
                utils.save_syn_clusters(config.syn_clustering_save_dir, assembly.idx, late_cluster_df, late_assembly=True)
            # some extra plotting
            cluster_df["rho"] = utils.get_syn_properties(c, cluster_df.index.to_numpy(), ["rho0_GB"])["rho0_GB"]
            cluster_dfs[assembly.idx[0]] = cluster_df
        fig_name = os.path.join(config.fig_path, "rho0_syn_clusts_%s.png" % seed)
        plot_cond_rhos(cluster_dfs, fig_name)

