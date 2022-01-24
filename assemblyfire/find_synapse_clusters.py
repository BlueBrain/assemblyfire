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
from assemblyfire.topology import AssemblyTopology
from assemblyfire.clustering import cluster_synapses
from assemblyfire.plots import plot_grouped_diffs

L = logging.getLogger("assemblyfire")


def run(config_path, debug):
    """
    Loads in asssemblies and connectivity matrix from saved h5 file, and for each assembly
    finds the most innervated L5_TTPCs, looks for synapse clusters and saved to pickle files
    :param config_path: str - path to project config file
    :param debug: bool - to save figures for visual inspection
    """

    config = Config(config_path)
    L.info(" Load in assemblies and connectivity matrix from %s" % config.h5f_name)
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)
    conn_mat = AssemblyTopology.from_h5(config.h5f_name,
                                        prefix=config.h5_prefix_connectivity, group_name="full_matrix")

    L.info(" Detecting synapse clusters and saving them to files")
    c = utils.get_bluepy_circuit(utils.get_sim_path(config.root_path).iloc[0])
    for seed, assembly_grp in tqdm(assembly_grp_dict.items(), desc="Iterating over seeds"):
        cluster_dfs = {}
        for assembly in tqdm(assembly_grp.assemblies, desc="%s syn. clusters" % seed, leave=False):
            fig_dir = os.path.join(config.fig_path, "%s_debug" % seed) if debug else None
            # sort gids by in-degree (in the assembly subgraph) and get first n L5_TTPCs
            sorted_gids = assembly.gids[np.argsort(conn_mat.degree(assembly, kind="in"))[::-1]]
            post_gids = sorted_gids[np.nonzero(c.cells.get(sorted_gids, "mtype").isin(["L5_TPC:A",
                        "L5_TPC:B"]).to_numpy())[0][:config.syn_clustering_n_neurons_sample]]
            # get clusters and save them to pickle
            cluster_df = cluster_synapses(c, post_gids, assembly, config.syn_clustering_target_range,
                                          config.syn_clustering_min_nsyns, fig_dir=fig_dir)
            utils.save_syn_clusters(config.root_path, assembly.idx, cluster_df)
            # some extra plotting
            cluster_df["rho"] = utils.get_syn_properties(c, cluster_df.index.to_numpy(), ["rho0_GB"])["rho0_GB"]
            cluster_dfs["assembly%i" % assembly.idx[0]] = cluster_df
        fig_name = os.path.join(config.fig_path, "rho0_syn_clusts_%s.png" % seed)
        plot_grouped_diffs(cluster_dfs, fig_name)

