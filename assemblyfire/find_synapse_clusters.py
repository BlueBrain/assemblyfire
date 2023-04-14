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
DSET_MEMBER = "member"
DSET_DEG = "degree"
DSET_CLST = "strength"
DSET_PVALUE = "pvalue"


def _get_degree_sorted_assembly_gids(c, conn_mat, assembly, mtype_list, n_samples, pre_assembly=None):
    """Helper function to select indegree sorted postsynaptic gids from assembly"""
    if pre_assembly is None:
        indegrees = conn_mat.degree(assembly.gids, kind="in")
    else:
        indegrees = conn_mat.degree(pre_gids=pre_assembly.gids, post_gids=assembly.gids, kind="in")
    sorted_assembly_gids = assembly.gids[np.argsort(indegrees)[::-1]]
    mtypes = utils.get_mtypes(c, sorted_assembly_gids)
    return mtypes.loc[mtypes.isin(mtype_list)].index.to_numpy()[:n_samples]


def _get_syn_nnd_degree_sorted_assembly_gids(c, syn_nnds, assembly, mtype_list, n_samples, p_th=0.05, pre_assembly=None):
    """Helper function to select indegree sorted postsynaptic gids from assembly
    (Compared to above there is a preselection og gids by significant synapse nnd. 'strength')"""
    assembly_id = "assembly%i" % pre_assembly.idx[0] if pre_assembly is not None else "assembly%i" % assembly.idx[0]
    df = syn_nnds.loc[:, [(assembly_id, DSET_MEMBER), (assembly_id, DSET_DEG),
                          (assembly_id, DSET_CLST), (assembly_id, DSET_PVALUE)]]
    df.columns = df.columns.get_level_values(1)
    # index out assembly members with significant syn nnd. 'strength', and sort them
    df = df.loc[(df[DSET_CLST] > 0) & (df[DSET_PVALUE] < p_th)]
    if pre_assembly is None:
        df = df.loc[df[DSET_MEMBER] == 1]
    else:
        df = df.iloc[np.in1d(df.index.to_numpy(), assembly.gids), :]
    df = df.sort_values(DSET_DEG, ascending=False)
    # index out mtypes, and take the first `n_samples`
    mtypes = utils.get_mtypes(c, df.index.to_numpy())
    return mtypes.loc[mtypes.isin(mtype_list)].index.to_numpy()[:n_samples]


def _get_cross_degree_sorted_assembly_gids(c, conn_mat, cross_assembly_grp, assembly, mtype_list, n_samples):
    """Similar indegree based helper as above, but works for cross-assembly connections
    (It'll return `n_samples` gids per presynaptic assembly (i.e. `len(cross_assembly_grp)`), not in total...)"""
    gids = [_get_degree_sorted_assembly_gids(c, conn_mat, assembly, mtype_list, n_samples, pre_assembly)
            for pre_assembly in cross_assembly_grp.assemblies]
    return np.unique(np.concatenate(gids))


def _get_cross_syn_nnd_degree_sorted_assembly_gids(c, syn_nnds, cross_assembly_grp, assembly, mtype_list, n_samples):
    """Similar synapse nnd. and indegree based helper as above, but works for cross-assembly connections
    (It'll return `n_samples` gids per presynaptic assembly (i.e. `len(cross_assembly_grp)`), not in total...)"""
    gids = [_get_syn_nnd_degree_sorted_assembly_gids(c, syn_nnds, assembly, mtype_list, n_samples, pre_assembly)
            for pre_assembly in cross_assembly_grp.assemblies]
    return np.unique(np.concatenate(gids))


def _update_cross_cluster_dfs_for_plotting(cross_cluster_dfs):
    """For each post assembly concatenates clustered vs. non-clustered results from all pre assemblies
    and calls it in a way that `plots.py/plot_cond_rhos()` can handle it..."""
    plot_cross_cluster_dfs = {}
    for post_assembly_id, cross_cluster_df in cross_cluster_dfs.items():
        df = cross_cluster_df.copy()
        pre_assemblies = [col_name for col_name in df.columns.to_numpy() if col_name.split("assembly")[0] == '']
        col_name = "assembly%i" % post_assembly_id
        df[col_name] = -100  # placeholder (see `clustering.py/cluster_synapses()`)
        # first overwrite all synapses that are coming from any assemblies
        df.loc[df["non_assembly"] == -100, col_name] = -1
        # then overwrite clusters (with a random cluster id: 100,
        # because plotting only cares if a synapse is part of a cluster but not the identity)
        for pre_assembly in pre_assemblies:
            df.loc[df[pre_assembly] >= 0, col_name] = 100
        # lastly delete all pre assembly columns (since we just made one for the combination of all)
        df.drop(columns=pre_assemblies, inplace=True)
        plot_cross_cluster_dfs[post_assembly_id] = df
    return plot_cross_cluster_dfs


def run(config_path, debug):
    """
    Loads in asssemblies from saved h5 file, and for each assembly samples gids from a given mtype (by indegree)
    looks for synapse clusters on them and saves them to pickle files
    :param config_path: str - path to project config file
    :param debug: bool - to save figures for visual inspection
    """

    config = Config(config_path)
    L.info(" Load in assemblies and connectivity matrix from %s" % config.h5f_name)
    sim_paths = utils.get_sim_path(config.root_path)
    c = utils.get_bluepy_circuit(sim_paths.iloc[0])
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)
    conn_mat = AssemblyTopology.from_h5(config.h5f_name,
                                        prefix=config.h5_prefix_connectivity, group_name="full_matrix")
    target_range, min_nsyns = config.syn_clustering_target_range, config.syn_clustering_min_nsyns
    mtypes, n_samples = config.syn_clustering_mtypes, config.syn_clustering_n_neurons_sample
    cross_assemblies = config.syn_clustering_cross_assemblies

    L.info(" Detecting synapse clusters and saving them to files")
    for seed, assembly_grp in tqdm(assembly_grp_dict.items(), desc="Iterating over seeds"):
        try:  # the utility function has an `assert` so it would just break if these are not calculated
            syn_nnds = utils.load_syn_nnd_from_h5(config.h5f_name, len(assembly_grp), prefix="%s_syn_nnd" % seed)
            L.info(" Using saved synapse nnds. (%i cells) for gid selection" % len(syn_nnds))
        except:
            syn_nnds = None
        cluster_dfs, cross_cluster_dfs = {}, {}
        for assembly in tqdm(assembly_grp.assemblies, desc="%s syn. clusters" % seed, leave=False):
            if syn_nnds is not None:
                gids = _get_syn_nnd_degree_sorted_assembly_gids(c, syn_nnds, assembly, mtypes, n_samples)
            else:
                gids = _get_degree_sorted_assembly_gids(c, conn_mat, assembly, mtypes, n_samples)
            loc_df = utils.get_synloc_df(c, utils.get_syn_idx(c.config["connectome"], conn_mat.gids, gids))
            # create a fake assembly "group" in order to look for *within* assembly clusters only
            single_assembly_grp = AssemblyGroup([assembly], all_gids=assembly.gids)
            # get clusters and save them to pickle
            if debug:
                fig_dir = os.path.join(config.fig_path, "%s_debug" % seed)
                utils.ensure_dir(fig_dir)
                cluster_df = cluster_synapses(loc_df, single_assembly_grp, target_range, min_nsyns,
                                              fig_dir=fig_dir, base_assembly_idx=assembly.idx[0], c=c)
            else:
                cluster_df = cluster_synapses(loc_df, single_assembly_grp, target_range, min_nsyns)
            utils.save_syn_clusters(config.syn_clustering_save_dir, assembly.idx, cluster_df)
            # some extra stuff for plotting
            cluster_df["rho"] = utils.get_syn_properties(c, cluster_df.index.to_numpy(), ["rho0_GB"])["rho0_GB"]
            cluster_dfs[assembly.idx[0]] = cluster_df
        fig_name = os.path.join(config.fig_path, "rho0_syn_clusts_%s.png" % seed)
        plot_cond_rhos(cluster_dfs, fig_name)

        if seed in cross_assemblies:
            # not optimal way to find postsynaptic cross assemblies, but better for progress bar...
            for assembly_id in tqdm(cross_assemblies[seed], desc="%s cross syn. clusters" % seed, leave=False):
                for assembly in assembly_grp.assemblies:
                    if assembly.idx[0] == assembly_id:
                        # create an assembly group with specified assemblies (based on temporal order) only
                        assembly_lst = [assembly for assembly in assembly_grp.assemblies
                                        if assembly.idx[0] in cross_assemblies[seed][assembly_id]]
                        all_gids = np.unique(np.concatenate([assembly.gids for assembly in assembly_lst]))
                        cross_assembly_grp = AssemblyGroup(assembly_lst, all_gids=all_gids)
                        # sample gids (slightly differently) to have high indegree from `cross_assembly_grp`

                        if syn_nnds is not None:
                            gids = _get_cross_syn_nnd_degree_sorted_assembly_gids(c, syn_nnds, cross_assembly_grp,
                                                                                  assembly, mtypes, n_samples)
                        else:
                            gids = _get_cross_degree_sorted_assembly_gids(c, conn_mat, cross_assembly_grp,
                                                                          assembly, mtypes, n_samples)
                        loc_df = utils.get_synloc_df(c, utils.get_syn_idx(c.config["connectome"], conn_mat.gids, gids))
                        if debug:
                            cross_cluster_df = cluster_synapses(loc_df, cross_assembly_grp, target_range, min_nsyns,
                                                                fig_dir=fig_dir, base_assembly_idx=assembly_id, c=c)
                        else:
                            cross_cluster_df = cluster_synapses(loc_df, cross_assembly_grp, target_range, min_nsyns)
                        utils.save_syn_clusters(config.syn_clustering_save_dir, assembly.idx, cross_cluster_df,
                                                cross_assembly=True)
                        cross_cluster_df["rho"] = utils.get_syn_properties(c, cross_cluster_df.index.to_numpy(),
                                                                           ["rho0_GB"])["rho0_GB"]
                        cross_cluster_dfs[assembly_id] = cross_cluster_df
            fig_name = os.path.join(config.fig_path, "rho0_cross_syn_clusts_%s.png" % seed)
            plot_cond_rhos(_update_cross_cluster_dfs_for_plotting(cross_cluster_dfs), fig_name)

