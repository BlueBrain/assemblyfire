"""
Consensus assemblies over time
(big mix of `../find_consensus_assemblies.py` and `consensus_botany.py`)
last modified: András Ecker 02.2022
"""

import os
import numpy as np

from assemblyfire.config import Config
import assemblyfire.utils as utils
from assemblyfire.clustering import cluster_assemblies
from assemblyfire.assemblies import ConsensusAssembly, build_assembly_group
from assemblyfire.topology import AssemblyTopology, simplex_counts_consensus_instantiations
from assemblyfire.plots import plot_assembly_sim_matrix, plot_dendogram_silhouettes,\
                               plot_consensus_mtypes, plot_simplex_counts_consensus


def _cluster_assemblies(assembly_grp, n_assemblies, fig_prefix, distance_metric, linkage_method):
    """Cluster assemblies and return consensus assembly dict"""
    sim_matrix, clusters, plotting = cluster_assemblies(assembly_grp.as_bool().T, n_assemblies,
                                                        distance_metric, linkage_method)
    plot_assembly_sim_matrix(sim_matrix, n_assemblies, fig_prefix + "simmat_%s.png" % distance_metric)
    plot_dendogram_silhouettes(clusters, *plotting, fig_prefix + "clustering_%s.png" % linkage_method)
    # making consensus assemblies from assemblies grouped by clustering
    consensus_assemblies = {}
    for cluster in np.unique(clusters):
        label = "cluster%i" % cluster
        c_idx = np.where(clusters == cluster)[0]
        assembly_lst = [assembly_grp.assemblies[i] for i in c_idx]
        consensus_assemblies[label] = ConsensusAssembly(assembly_lst, index=cluster, label=label)
    return consensus_assemblies


def _consensus_botany(consensus_assemblies, conn_mat, all_gids, mtypes, ystuff, depths, fig_prefix):
    """Plots simplex counts and unions' vs. cores' depth profile and mtype composition"""
    simplex_counts, simplex_counts_control = simplex_counts_consensus_instantiations(consensus_assemblies, conn_mat)
    plot_simplex_counts_consensus(simplex_counts, simplex_counts_control, fig_prefix + "simplex_counts.png")

    consensus_gids = [assembly.gids for _, assembly in consensus_assemblies.items()]
    union_gids = [assembly.union.gids for _, assembly in consensus_assemblies.items()]
    consensus_mtypes = [mtypes[np.searchsorted(all_gids, gids)] for gids in consensus_gids]
    union_mtypes = [mtypes[np.searchsorted(all_gids, gids)] for gids in union_gids]
    plot_consensus_mtypes(union_gids, union_mtypes, consensus_gids, all_gids, consensus_mtypes, mtypes,
                          ystuff, depths, fig_prefix + "cons_mtypes.png")


def consensus_over_time_hc(assembly_grp_dict, conn_mat, t_chunk_idx, ystuff, depths, fig_path,
                           backward=True, distance_metric="jaccard", linkage_method="ward"):
    """
    Reimplementation of `assemblies.py/consensus_over_seed_hc()` by starting from 3 temporal chunks and adding
    more assemblies in every step. (It has way too many inputs because of plotting...)
    """
    t_chunk_idx = t_chunk_idx[::-1] if backward else t_chunk_idx  # invert time if needed
    t_chunk_labels = ["seed%i" % t for t in t_chunk_idx]
    # base 3 temporal chunks (from 2 it's kind of useless to make a consensus)
    gids, n_assemblies, assembly_lst = [], [], []
    gids, n_assemblies, assembly_lst, assembly_grp = build_assembly_group(gids, n_assemblies, assembly_lst,
                                                                          t_chunk_labels[:3], assembly_grp_dict)
    fig_prefix = os.path.join(fig_path, "consensus_over_time", "t%i-%i_" % (t_chunk_idx[0], t_chunk_idx[2]))
    consensus_assemblies = _cluster_assemblies(assembly_grp, n_assemblies, fig_prefix, distance_metric, linkage_method)
    all_gids, mtypes = conn_mat.gids, conn_mat.mtype
    _consensus_botany(consensus_assemblies, conn_mat, all_gids, mtypes, ystuff, depths, fig_prefix)
    # main loop that adds more temporal chunks step-by-step
    for t_chunk_id, t_chunk_label in zip(t_chunk_idx[3:], t_chunk_labels[3:]):
        gids, n_assemblies, assembly_lst, assembly_grp = build_assembly_group(gids, n_assemblies, assembly_lst,
                                                                              [t_chunk_label], assembly_grp_dict)
        fig_prefix = os.path.join(fig_path, "consensus_over_time", "t%i-%i_" % (t_chunk_idx[0], t_chunk_id))
        consensus_assemblies = _cluster_assemblies(assembly_grp, n_assemblies, fig_prefix, distance_metric, linkage_method)
        _consensus_botany(consensus_assemblies, conn_mat, all_gids, mtypes, ystuff, depths, fig_prefix)


def main(config_path):
    # preload stuff that will remain the same
    config = Config(config_path)
    project_metadata = utils.read_base_h5_metadata(config.h5f_name)
    t_chunks, t_chunk_idx = project_metadata["t"], project_metadata["seeds"]
    if len(t_chunks) == 2:
        raise RuntimeError("This script only works for chunked simulations!")
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)
    conn_mat = AssemblyTopology.from_h5(config.h5f_name, prefix=config.h5_prefix_connectivity)
    depths, ystuff = utils.get_figure_asthetics(utils.get_sim_path(config.root_path).iloc[0], config.target)
    # main run function...
    consensus_over_time_hc(assembly_grp_dict, conn_mat, t_chunk_idx, ystuff, depths, config.fig_path)


if __name__ == "__main__":
    config_path = "../configs/v7_10mins.yaml"
    main(config_path)
