"""
Consensus assembly botany
last modified: András Ecker 02.2023
"""

import os
import numpy as np
import pandas as pd

from assemblyfire.config import Config
import assemblyfire.utils as utils
from assemblyfire.topology import AssemblyTopology, simplex_counts_consensus_instantiations
import assemblyfire.plots as plots


def _create_cons_inst_lookup(consensus_assemblies):
    """Creates lookup table from consensus assembly instantiations (for easier mapping later)"""
    return {i: [assembly.idx for assembly in consensus_assemblies["cluster%i" % i].instantiations]
            for i in range(len(consensus_assemblies))}


def _find_cons_cluster_id(assembly_id, consensus_instantiations):
    """Finds assembly id in consensus instantiations"""
    for consensus_cluster_id, instantiations in consensus_instantiations.items():
        if assembly_id in instantiations:
            return consensus_cluster_id


def consensus_cluster_seqs(assembly_grp_dict, consensus_assemblies, metadata, fig_path):
    """Re-colors cluster (of time bin) sequences based on the IDs of the consensus assemblies"""
    consensus_instantiations = _create_cons_inst_lookup(consensus_assemblies)
    n_clusters = len(consensus_instantiations)
    stim_times, patterns = metadata["stim_times"], metadata["patterns"]

    all_clusters = {}
    for seed, asassembly_grp in assembly_grp_dict.items():
        orig_clusters, t_bins = metadata["clusters"][seed], metadata["t_bins"][seed]
        clusters = -1 * np.ones_like(orig_clusters)
        for assembly in asassembly_grp.assemblies:
            clusters[orig_clusters == assembly.idx[0]] = _find_cons_cluster_id(assembly.idx, consensus_instantiations)
        all_clusters[seed] = clusters

        fig_name = os.path.join(fig_path, "consensus_cluster_seq_%s.png" % seed)
        plots.plot_cons_cluster_seqs(clusters, t_bins, stim_times, patterns, n_clusters, fig_name)
    fig_name = os.path.join(fig_path, "consensus_cluster_seqs_all_seeds.png")
    plots.plot_cons_cluster_seqs_all_seeds(all_clusters, metadata["t_bins"], stim_times, patterns, n_clusters, fig_name)


def consensus_botany(assembly_grp, conn_mat, fig_path):
    """Plots cores' location, unions' and cores' mtype composition,
    and simplex counts of instantiations and their consensus' core's"""

    nrn_df = conn_mat.vertices
    gids, mtypes = nrn_df["gid"].to_numpy(), nrn_df["mtype"].to_numpy()
    core_idx = [assembly.idx[0] for assembly in assembly_grp.assemblies]  # just to work with the plotting fn.
    fig_name = os.path.join(fig_path, "consensus_assemblies.png")
    plots.plot_assemblies(assembly_grp.as_bool(), core_idx, assembly_grp.all, nrn_df.set_index("gid"), fig_name)

    core_mtypes = [mtypes[np.in1d(gids, assembly.gids, assume_unique=True)] for assembly in assembly_grp.assemblies]
    union_mtypes = [mtypes[np.in1d(gids, assembly.union.gids, assume_unique=True)] for assembly in assembly_grp.assemblies]
    plots.plot_consensus_mtypes(mtypes, core_mtypes, union_mtypes, os.path.join(fig_path, "consensus_mtypes.png"))

    simplex_counts, simplex_counts_control = simplex_counts_consensus_instantiations(assembly_grp, conn_mat)
    fig_name = os.path.join(fig_path, "simplex_counts_consensus.png")
    plots.plot_simplex_counts_consensus(simplex_counts, simplex_counts_control, fig_name)


def consensus_vs_single_cell_features(consensus_assemblies, single_cell_features, fig_path):
    """Plots the distributions of spike time reliability in consensus assemblies"""
    all_gids, r_spikes = single_cell_features["gids"], single_cell_features["r_spikes"]

    dfs, cons_assembly_gids = [], []
    for cons_assembly_id, assembly in consensus_assemblies.items():
        gids = assembly.gids
        df = pd.DataFrame(data=r_spikes[np.in1d(all_gids, gids)], index=gids, columns=["r_spike"], dtype=np.float32)
        df["consensus assembly id"] = cons_assembly_id.split("cluster")[1]
        dfs.append(df)
        cons_assembly_gids.extend(gids)
    gids = np.unique(cons_assembly_gids)
    idx = ~np.in1d(all_gids, gids)
    df = pd.DataFrame(data=r_spikes[idx], index=all_gids[idx], columns=["r_spike"], dtype=np.float32)
    df["consensus assembly id"] = "non assembly"
    dfs.append(df)
    df = pd.concat(dfs, join="inner")  # inner join will keep gids that are part of multiple assemblies
    plots.plot_consensus_r_spikes(df, os.path.join(fig_path, "consensus_r_spikes.png"))


def main(config_path):
    config = Config(config_path)
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)
    consensus_assemblies = utils.load_consensus_assemblies_from_h5(config.h5f_name, config.h5_prefix_consensus_assemblies)
    conn_mat = AssemblyTopology.from_h5(config.h5f_name, prefix=config.h5_prefix_connectivity)
    fig_path = config.fig_path

    consensus_cluster_seqs(assembly_grp_dict, consensus_assemblies, utils.read_cluster_seq_data(config.h5f_name), fig_path)
    consensus_botany(utils.consensus_dict2assembly_grp(consensus_assemblies), conn_mat, fig_path)

    if config.t_chunks is None:  # no single cell features for consensus over time
        single_cell_features = utils.load_single_cell_features_from_h5(config.h5f_name, config.h5_prefix_single_cell)
        consensus_vs_single_cell_features(consensus_assemblies, single_cell_features, fig_path)


if __name__ == "__main__":
    main("../configs/np_10seeds.yaml")
