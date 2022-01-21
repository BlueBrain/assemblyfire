"""
...
last modified: András Ecker 01.2022
"""

import os
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.spatial.distance import pdist, squareform

import assemblyfire.utils as utils
from assemblyfire.config import Config
from assemblyfire.topology import AssemblyTopology


def _create_lookups(syn_df, assembly):
    """Create dicts with synapse idx and fraction of those (compared to total)
    atm. it's only 1 assembly vs. the rest but can be extended (and the rest of the code will handle the extension)"""
    from bluepy.enums import Synapse
    assembly_syns = syn_df[Synapse.PRE_GID].isin(assembly.gids).to_numpy()
    assembly_frac = sum(assembly_syns) / len(assembly_syns)
    syn_idx = {"assembly%i" % assembly.idx[0]: np.nonzero(assembly_syns)[0],
               "non_assembly": np.nonzero(np.invert(assembly_syns))[0]}
    fracs = {"assembly%i" % assembly.idx[0]: assembly_frac, "non_assembly": 1 - assembly_frac}
    return syn_idx, fracs


def syn_distances(syn_df, mask_col, xzy_cols):
    """Return (Euclidean) distance between synapses on the same section (the rest is masked with nans)"""
    dists = squareform(pdist(syn_df[xzy_cols].to_numpy()))
    mask = squareform(pdist(syn_df[mask_col].to_numpy().reshape(-1, 1)))
    dists[mask > 0] = np.nan
    np.fill_diagonal(dists, np.nan)
    return dists


def distance_model(dists, fracs, target_range, fig_name=None):
    """Creates a cumulative histogram of (valid) inter-synapse distances, fits a line to it
    and based on the slope returns and underlying Poisson model (the mathematical assumption behind the Poisson is
    that the distribution should be uniform (histogram should be flat) and the cumulative a straight line)
    use fig_name != None to save a figure and visually verify"""
    dists = squareform(dists, checks=False)  # convert back to condensed form to not count every distance twice
    dist_samples = dists[~np.isnan(dists)]
    d_bins = np.arange(2.0 * target_range)  # *2 is totally arbitrary... it's just bigger than the target range
    hist, _ = np.histogram(dist_samples, d_bins)
    cum = np.cumsum(hist) / dists.shape[0]
    fit = np.polyfit(d_bins[1:], cum, 1)
    slope = fit[0]
    if fig_name is not None:
        from assemblyfire.plots import plot_synapse_distance_dist
        plot_synapse_distance_dist(d_bins, hist, cum, fit, fig_name)
    return {label: poisson(target_range * slope * frac) for label, frac in fracs.items()}


def merge_clusters(clusters):
    """Cleans raw clusters and merges them by taking the boolean array of raw clusters,
    and for each cluster `i` (while loop) checks all cluster `j`s for common synapses (`np.any()`, no explicit loop)
    and if there are any, adds those to cluster `i` and deletes cluster `j`.
    (This is needed as all clusters are detected minimum `min_nsyns` times at the first place.
    Input `cluster.shape[1]` is the number of raw clusters,
    while output `cluster.shape[1]` is the number of merged clusters)
    """
    i = 0
    while i < (clusters.shape[1] - 1):
        idx_partners = np.arange(i + 1, clusters.shape[1])
        matches = np.any(clusters[:, [i]] & clusters[:, idx_partners], axis=0)
        if not np.any(matches):
            i += 1
        else:
            # idx_partners[matches] are the column indices (`j`s) where cluster `i` has at least 1 common synapse
            clusters[:, i] = np.any(clusters[:, [i] + idx_partners[matches].tolist()], axis=1)
            clusters = clusters[:, ~np.in1d(range(clusters.shape[1]), idx_partners[matches])]
    return clusters


def find_clusters(dists, syn_idx_dict, target_range, min_nsyns, model_dict, log_sign_th=5.0):
    """
    Finds `min_nsyns` sized clusters of synapses within `target_range` (um) and tests their significance
    against the Poisson models build above in `distance_model()`
    :param dists: square matrix with distances between synapses (on the same section)
    :param syn_idx_dict: idx (in `dists`) of synapses belonging to given classes (see `_create_lookups()` above)
    :param target_range: max distance from synapse center to consider for clustering
    :param min_nsyns: minimum number of synapses within one cluster
    :param model_dict: control models for significance test (see `distance_model` above)
    :param log_sign_th: significance threshold (after taking log of p-values)
    :return: results: numpy array with one per synapse and one column per label
                      placeholder: -100, synapse belonging to the label: -1, cluster idx start at 0
             labels: keys from `syn_idx_dict` and `model_dict` (see `_create_lookups()` above)
    """
    labels = list(syn_idx_dict.keys())
    results = -100 * np.ones((dists.shape[0], len(labels)), dtype=int)
    for i, label in enumerate(labels):
        syn_idx = syn_idx_dict[label]
        results[syn_idx, i] = -1
        sub_dists = dists[np.ix_(syn_idx, syn_idx)]
        nsyns = (sub_dists < target_range).sum(axis=1)
        p_vals = 1.0 - model_dict[label].cdf(nsyns - 1)
        p_vals[p_vals == 0.0] += 1 / np.power(10, 2*log_sign_th)  # for numerical stability (see log10 in the next line)
        significant = (-np.log10(p_vals) >= log_sign_th) & (nsyns >= min_nsyns)
        if np.any(significant):
            raw_clusters = sub_dists[:, significant] < target_range
            merged_clusters = merge_clusters(raw_clusters)
            row_idx, col_idx = np.nonzero(merged_clusters)
            results[syn_idx[row_idx], i] = col_idx  # set cluster labels (starting at 0)
    return results, labels


def assembly_synapse_clustering(config_path, n_sample=10, target_range=10.0, min_nsyns=5, debug=False):
    """Loads in assemblies and, sample `n_sample` postsynaptic neurons (ordered by indegree) and in those
    looks for synapse clusters (of at least minimum `min_nsyns` synapses within `target_range` um)"""

    config = Config(config_path)
    sim_paths = utils.get_sim_path(config.root_path)
    c = utils.get_bluepy_circuit(sim_paths.iloc[0])
    gids = utils.get_E_gids(c, "hex_O1")
    conn_mat = AssemblyTopology.from_h5(config.h5f_name,
                                        prefix=config.h5_prefix_connectivity, group_name="full_matrix")
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)
    # putting this here in order to get an ImportError from `utils.get_bluepy_circuit()` if bluepy is not installed...
    from bluepy.enums import Synapse
    xyz = [Synapse.POST_X_CENTER, Synapse.POST_Y_CENTER, Synapse.POST_Z_CENTER]
    syn_properties = [Synapse.PRE_GID, Synapse.POST_GID, Synapse.POST_SECTION_ID] + xyz

    for seed, assembly_grp in assembly_grp_dict.items():
        for assembly in tqdm(assembly_grp.assemblies, desc="%s syn. clusters" % seed):
            if debug:
                fig_dir = os.path.join(config.fig_path, "%s_debug" % seed)
                utils.ensure_dir(fig_dir)
            in_degrees = conn_mat.degree(assembly, kind="in")
            sort_idx = np.argsort(in_degrees)[::-1]
            post_gids = assembly.gids[sort_idx[:n_sample]]
            syn_df = utils.get_syn_properties(c, utils.get_syn_idx(c, gids, post_gids), syn_properties)
            cluster_dfs = []
            for gid in post_gids:
                syn_df_gid = syn_df.loc[syn_df[Synapse.POST_GID] == gid]
                syn_idx, fracs = _create_lookups(syn_df_gid, assembly)
                dists = syn_distances(syn_df_gid, Synapse.POST_SECTION_ID, xyz)
                if debug:
                    fig_name = os.path.join(fig_dir, "assembly%i_a%i_synapse_dists.png" % (assembly.idx[0], gid))
                    model = distance_model(deepcopy(dists), fracs, target_range, fig_name=fig_name)
                else:
                    model = distance_model(deepcopy(dists), fracs, target_range)
                clusters, labels = find_clusters(dists, syn_idx, target_range, min_nsyns, model)
                data = np.concatenate((syn_df_gid[[Synapse.PRE_GID, Synapse.POST_GID]].to_numpy(), clusters), axis=1)
                cluster_df = pd.DataFrame(data=data, index=syn_df_gid.index, columns=["pre_gid", "post_gid"] + labels)
                if debug:
                    from assemblyfire.plots import plot_synapse_clusters
                    morph = c.morph.get(int(gid), transform=True)
                    fig_name = os.path.join(fig_dir, "assembly%i_a%i_synapse_clusters.png" % (assembly.idx[0], gid))
                    plot_synapse_clusters(morph, pd.concat((cluster_df[labels], syn_df_gid[xyz]), axis=1), xyz, fig_name)
                cluster_dfs.append(cluster_df)
            cluster_df = pd.concat(cluster_dfs)
            utils.save_syn_clusters(config.root_path, assembly.idx, cluster_df)


if __name__ == "__main__":
    config_path = "../configs/v7_10seeds.yaml"
    assembly_synapse_clustering(config_path)
