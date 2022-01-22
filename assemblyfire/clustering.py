"""
Functions to run either hierarchical clustering (via Ward's linkage)
of the cosine similarity matrix of significant time bins
a la Perez-Ortega et al. 2020 (see also Carillo-Reid et al. 2015), or
density based clustering a la Rodriguez and Laio 2014
(slightly modified by Yger et al. 2018)
of spike matrix projected to PCA space a la Herzog et al. 2020.
Then "core-cells" and cell assemblies are detected with correlation
based methods from (Montijn et al. 2016 and) Herzog et al. 2020
Assemblies are clustered into consensus assemblies via hierarchical clustering
last modified: András Ecker 01.2022
"""

import os
import logging
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import multiprocessing as mp
from scipy.optimize import curve_fit
from scipy.stats import poisson
from scipy.stats.distributions import t
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score, silhouette_samples, davies_bouldin_score

L = logging.getLogger("assemblyfire")


def cosine_similarity(X):
    """Cosine similarity matrix calculation (using only core numpy)
    much faster then `1-squareform(pdist(X, metrix="cosine"))`"""
    X_norm = X / np.linalg.norm(X, axis=-1)[:, np.newaxis]
    return np.dot(X_norm, X_norm.T)


# hierarchical clustering (using scipy and sklearn)
def cluster_sim_mat(spike_matrix, min_n_clusts=5, max_n_clusts=20, n_method="DB"):
    """Hieararchical (Ward linkage) clustering of cosine similarity matrix of significant time bins"""

    sim_matrix = cosine_similarity(spike_matrix.T)
    dists = 1 - sim_matrix
    dists[dists < 1e-10] = 0.  # fixing numerical errors
    cond_dists = squareform(dists)  # squareform implements its inverse if the input is a square matrix

    linkage_matrix = linkage(cond_dists, method="ward")

    # determine number of clusters using silhouette scores or Davies-Bouldin index
    silhouette_scores, DB_scores = [], []
    for n in range(min_n_clusts, max_n_clusts+1):
        clusters = fcluster(linkage_matrix, n, criterion="maxclust")
        silhouette_scores.append(silhouette_score(dists, clusters))
        DB_scores.append(davies_bouldin_score(dists, clusters))
    assert n_method in ["ss", "DB"], "Only silhouette scores and Davies-Bouldin index are supported atm."
    if n_method == "ss":
        n_clust = np.argmax(silhouette_scores) + min_n_clusts
    elif n_method == "DB":
        n_clust = np.argmin(DB_scores) + min_n_clusts

    clusters = fcluster(linkage_matrix, int(n_clust), criterion="maxclust")
    silhouettes = silhouette_samples(dists, clusters) if n_method == "ss" else None

    plotting = [linkage_matrix, silhouettes]
    return sim_matrix, clusters - 1, plotting


# density based clustering (built from core numpy and scipy functions)
def PCA_ncomps(matrix, n_components):
    """PCA wrapper with fixed number of components"""
    F = PCA(n_components)
    transformed = F.fit_transform(matrix)
    return transformed, F.components_


def calc_rho_delta(dists, ratio_to_keep):
    """Herzog et al. 2020: calculates local density: \rho_i = 1 / (1/N*\sum_{j:d_ij<d_c} d_ij) and
    minimum distance with points with higher density: \delta_i = min_{j:\rho_j>\rho_i} (d_ij)"""
    # keep a given % of the neighbours a la Yger et al. 2018
    # (not constant d_c as in the original Rodrigez and Laio 2014 paper)
    n2keep = int(ratio_to_keep * dists.shape[1])
    # taking the mean distance of the closest neighbours
    mean_min_dists = np.mean(np.sort(dists, axis=1)[:, 0:n2keep], axis=1)
    # replacing 0s as dividing by 0 would give an error in the next line
    mean_min_dists[mean_min_dists == 0] = np.min(mean_min_dists[mean_min_dists != 0])
    # inverse of mean minimum distances -> density
    rhos = 1 / mean_min_dists
    max_rho = np.max(rhos)

    deltas = np.zeros_like(rhos, dtype=np.float)
    for i, rho in enumerate(rhos):
        if rho != max_rho:
            idx = np.where(rhos > rho)[0]
            deltas[i] = np.min(dists[i, idx])
        else:
            deltas[i] = np.max(dists[i, :])
    return rhos, deltas


def _fn(x, m, yshift):
    """Dummy function to be passed to `cure_fit()``"""
    return x*m + yshift


def fit_gammas(gammas, alpha=0.001):
    """Fits sorted gammas"""
    tmp_idx = np.arange(0, len(gammas))
    popt, pcov = curve_fit(_fn, tmp_idx, gammas)
    dof = len(gammas)-len(popt)
    t_val = t.ppf(1-alpha/2, dof)
    stds = t_val * np.sqrt(np.diagonal(pcov))
    return tmp_idx, popt, stds


def db_clustering(matrix, ratio_to_keep=0.02):
    """Density based clustering a la Rodriguez and Laio 2014"""
    # calc pairwise_distances
    dists = squareform(pdist(matrix, metric="euclidean"))
    np.fill_diagonal(dists, np.max(dists))  # get rid of zero dists in the diagonal
    # define density
    rhos, deltas = calc_rho_delta(dists, ratio_to_keep)
    gammas = rhos * deltas
    sort_idx = np.argsort(gammas, kind="mergsort")[::-1]
    sorted_gammas = gammas[sort_idx]
    # fit gammas to define threshold
    tmp_idx, popt, stds = fit_gammas(sorted_gammas)
    fit = _fn(tmp_idx, *popt)
    lCI = _fn(tmp_idx, *(popt-stds))
    uCI = _fn(tmp_idx, *(popt+stds))
    # find cluster centroids
    centroid_idx = np.where(gammas >= uCI)[0]
    assert len(centroid_idx) <= 20, "len(centroid_idx)=%i > 20" % len(centroid_idx)
    plotting = [rhos, deltas, tmp_idx, sorted_gammas, fit, lCI, uCI, centroid_idx]
    # assign points to centroids
    clusters = np.argmin(dists[:, centroid_idx], axis=1)
    clusters[centroid_idx] = np.arange(len(centroid_idx))
    return clusters, plotting


# core-cells and assemblies related functions (mostly calculating correlations)
def pairwise_correlation_X(x):
    """Pairwise correlation between rows of a matrix x"""
    corrs = 1 - pairwise_distances(x, metric="correlation", n_jobs=-1, force_all_finite=False)
    corrs[np.isnan(corrs)] = 0.
    return corrs


def pairwise_correlation_XY(x, y):
    """Pairwise correlation between rows of a matrix X and matrix Y"""
    corrs = 1 - cdist(x, y, metric="correlation")
    corrs[np.isnan(corrs)] = 0.
    return corrs


def _convert_clusters(clusters):
    """Convert clusters into a sparse matrix form for `cdist()`"""
    sparse_clusters = np.zeros((len(np.unique(clusters)), clusters.shape[0]), dtype=int)
    for i in np.unique(clusters):
        sparse_clusters[i, clusters == i] = 1
    return sparse_clusters


def corr_spike_matrix_clusters(spike_matrix, sparse_clusters):
    """Correlation of cells with clusters (of time bins)"""
    return pairwise_correlation_XY(spike_matrix, sparse_clusters)


def corr_shuffled_spike_matrix_clusters(spike_matrix, sparse_clusters):
    """Random shuffles columns of the spike matrix (keeps number of spikes per neuron)
    in order to create surrogate dataset for significance test of correlation"""
    spike_matrix_rnd = spike_matrix[:, np.random.permutation(spike_matrix.shape[1])]
    return corr_spike_matrix_clusters(spike_matrix_rnd, sparse_clusters)


def _corr_spikes_clusters_subprocess(inputs):
    """Subprocess used by multiprocessing pool for setting significance threshold"""
    return corr_shuffled_spike_matrix_clusters(*inputs)


def sign_corr_ths(spike_matrix, sparse_clusters, N=1000):
    """Generates surrogate datasets and calculates correlation coefficients
    then takes 95% percentile of the surrogate datasets as a significance threshold"""
    n = N if mp.cpu_count()-1 > N else mp.cpu_count()-1
    pool = mp.Pool(processes=n)
    corrs = pool.map(_corr_spikes_clusters_subprocess, zip([spike_matrix for _ in range(N)],
                                                           [sparse_clusters for _ in range(N)]))
    pool.terminate()
    corrs = np.dstack(corrs)  # shape: ngids x nclusters x N
    # get sign threshold (compare to Monte-Carlo shuffles)
    corr_ths = np.percentile(corrs, 95, axis=2, overwrite_input=True)
    return corr_ths


def within_cluster_correlations(spike_matrix, core_cell_idx):
    """Compares within cluster correlations (correlation of core cells)
    against the avg. correlation in the whole dataset
    if the within cluster correlation it's higher the cluster is an assembly"""
    corrs = pairwise_correlation_X(spike_matrix)
    np.fill_diagonal(corrs, np.nan)
    mean_corr = np.nanmean(corrs)

    assembly_idx = []
    for i in range(core_cell_idx.shape[1]):
        idx = np.where(core_cell_idx[:, i] == 1)[0]
        if np.nanmean(corrs[np.ix_(idx, idx)]) > mean_corr:
            assembly_idx.append(i)
    return assembly_idx


def cluster_spikes(spike_matrix_dict, method, overwrite_seeds, FigureArgs):
    """
    Cluster spikes either via hierarchical clustering (Ward's linkage)
    of the cosine similarity matrix of significant time bins (see Perez-Ortega et al. 2020), or
    density based clustering of spike matrix projected to PCA space (see Herzog et al. 2020)
    :param spike_matrix_dict: dict with seed as key and SpikeMatrixResult (see `spikes.py`) as value
    :param method: str - clustering method (read from yaml config file)
    :param overwrite_seeds: dict with seeds as keys and values as desired number of clusters
                           (instead of the optimal cluster number determined with the current heuristics in place)
    :param FigureArgs: plotting related arguments (see `find_assemblies.py`)
    :return: dict with seed as key and clustered (significant) time bins as value
    """
    from assemblyfire.plots import plot_cluster_seqs, plot_pattern_clusters

    fig_path = FigureArgs.fig_path
    ts, stim_times, patterns,  = FigureArgs.t, FigureArgs.stim_times, np.asarray(FigureArgs.patterns)
    clusters_dict = {}
    for seed, SpikeMatrixResult in tqdm(spike_matrix_dict.items(), desc="Clustering"):
        spike_matrix, t_bins = SpikeMatrixResult.spike_matrix, SpikeMatrixResult.t_bins
        if len(ts) > 2:  # if t is chunked the chunk bounds are saved and "seeds" represent chunks
            idx = np.where((ts[seed] <= stim_times) & (stim_times < ts[seed + 1]))[0]
        else:
            idx = np.arange(len(stim_times))  # just to not break the code...

        if method == "hierarchical":
            from assemblyfire.plots import plot_sim_matrix, plot_dendogram_silhouettes
            if seed not in overwrite_seeds:
                sim_matrix, clusters, plotting = cluster_sim_mat(spike_matrix)
            else:
                sim_matrix, clusters, plotting = cluster_sim_mat(spike_matrix, min_n_clusts=overwrite_seeds[seed],
                                                                 max_n_clusts=overwrite_seeds[seed])
            fig_name = os.path.join(fig_path, "similarity_matrix_seed%i.png" % seed)
            plot_sim_matrix(sim_matrix, t_bins, stim_times[idx], patterns[idx], fig_name)
            fig_name = os.path.join(fig_path, "ward_clustering_seed%i.png" % seed)
            plot_dendogram_silhouettes(clusters, *plotting, fig_name)
        elif method == "density_based":
            from assemblyfire.plots import plot_transformed, plot_components, plot_rhos_deltas
            pca_transformed, pca_components = PCA_ncomps(spike_matrix.T, 12)
            # TODO: add overwrite_seeds here as well
            clusters, plotting = db_clustering(pca_transformed)

            fig_name = os.path.join(fig_path, "PCA_transformed_seed%i.png" % seed)
            plot_transformed(pca_transformed, t_bins, stim_times[idx], patterns[idx], fig_name)
            fig_name = os.path.join(fig_path, "PCA_components_seed%i.png" % seed)
            plot_components(pca_components, SpikeMatrixResult.gids, FigureArgs.depths, fig_name)
            fig_name = os.path.join(fig_path, "rho_delta_seed%i.png" % seed)
            plot_rhos_deltas(*plotting, fig_name)

        clusters_dict[seed] = clusters
        fig_name = os.path.join(fig_path, "cluster_seq_seed%i.png" % seed)
        plot_cluster_seqs(clusters, t_bins, stim_times[idx], patterns[idx], fig_name)
        fig_name = os.path.join(fig_path, "clusters_patterns_seed%i.png" % seed)
        plot_pattern_clusters(clusters, t_bins, stim_times[idx], patterns[idx], fig_name)
    return clusters_dict


def detect_assemblies(spike_matrix_dict, clusters_dict, h5f_name, h5_prefix, FigureArgs):
    """
    Finds "core cells" - cells which correlate with the activation of (clustered) time bins
    and then checks within group correlations against the mean correlation to decide
    if the core cell group is an assembly or not a la Herzog et al. 2020
    :param spike_matrix_dict: dict with seed as key and SpikeMatrixResult (see `spikes.py`) as value
    :param clusters_dict: dict with seed as key and clustered (significant) time bins as value (see `cluster_spikes()`)
    :param h5f_name: str - name of the HDF5 file (dumping the assemblies and their metadata)
    :param h5_prefix: str - directory name of assemblies within the HDF5 file
    :param FigureArgs: plotting related arguments (see `cli.py`)
    """
    from assemblyfire.assemblies import Assembly, AssemblyGroup
    from assemblyfire.plots import plot_assemblies

    for seed, SpikeMatrixResult in tqdm(spike_matrix_dict.items(), desc="Detecting assemblies"):
        spike_matrix = SpikeMatrixResult.spike_matrix
        gids = SpikeMatrixResult.gids
        clusters = clusters_dict[seed]

        # core cells
        sparse_clusters = _convert_clusters(clusters)
        corrs = corr_spike_matrix_clusters(spike_matrix, sparse_clusters)
        corr_ths = sign_corr_ths(spike_matrix, sparse_clusters)
        core_cell_idx = np.zeros_like(corrs, dtype=int)
        core_cell_idx[np.where(corrs > corr_ths)] = 1
        # cell assemblies
        assembly_idx = within_cluster_correlations(spike_matrix, core_cell_idx)

        # save to h5
        metadata = {"clusters": clusters}
        assembly_lst = [Assembly(gids[core_cell_idx[:, i] == 1], index=(i, seed)) for i in assembly_idx]
        assemblies = AssemblyGroup(assemblies=assembly_lst, all_gids=gids, label="seed%i" % seed, metadata=metadata)
        assemblies.to_h5(h5f_name, prefix=h5_prefix)

        # plot (only depth profile at this point)
        fig_name = os.path.join(FigureArgs.fig_path, "assemblies_seed%i.png" % seed)
        plot_assemblies(core_cell_idx, assembly_idx, gids,
                        FigureArgs.ystuff, FigureArgs.depths, fig_name)


def _check_seed_separation(clusters, n_assemblies_cum):
    """TODO"""
    for i, j in zip(n_assemblies_cum[:-1], n_assemblies_cum[1:]):
        _, counts = np.unique(clusters[i:j], return_counts=True)
        if (counts > 1).any():
            return False
    return True


def _update_block_diagonal_dists(dists, n_assemblies_cum):
    """
    Assemblies from the same seed tend to cluster together, but that's not what we want. - Daniela
    Thus, this function fills block diagonals with "infinite" distance representing infinite distance
    between different assemblies from the same seed and return scipy's condensed distance representation
    which can be passed to hierarhichal clustering in the next step
    """
    inf_dist = np.max(dists) * 5
    for i, j in zip(n_assemblies_cum[:-1], n_assemblies_cum[1:]):
        dists[i:j, i:j] = inf_dist
    np.fill_diagonal(dists, 0)
    return squareform(dists)


def cluster_assemblies(assemblies, n_assemblies, distance_metric, linkage_method,
                       update_block_diagonals=False, n_method="min"):
    """
    Hieararchical (Ward linkage) clustering of hamming similarity matrix of assemblies from different seeds
    :param assemblies: assemblies x gids boolean array representing all assemblies across seeds
    :param n_assemblies: list with number of assemblies per seed
    :param distance_metric: distance metrics to use on assemblies (has to be valid in scipy's `pdist()`)
    :param linkage_method: linkage method to use (has to be valid in scipy's `hierarchy.linakge()`)
    :param update_block_diagonals: see `_update_block_diagonal_dists()` above
    :param n_method: method to determine optimal cluster number
    """
    cond_dists = pdist(assemblies, metric=distance_metric)
    dists = squareform(cond_dists)
    sim_matrix = 1 - dists
    # update block diagonals of the distance matrix to prevent assemblies from the same seed to cluster together
    n_assemblies_cum = [0] + np.cumsum(n_assemblies).tolist()
    if update_block_diagonals:
        cond_dists = _update_block_diagonal_dists(dists, n_assemblies_cum)
    # determine n_cluster range: min: max nr. of assemblies in one seed (if it was lower they would cluster together)
    # max: max nr. of assemblies or hard coded 20
    min_n_clusts = np.max(n_assemblies)
    max_n_clusts = 20 if n_assemblies_cum[-1] >= 20 else n_assemblies_cum[-1]

    linkage_matrix = linkage(cond_dists, method=linkage_method)

    # determine number of clusters using the combination of silhouette scores or Davies-Bouldin index
    # and the fact that we don't want assemblies from the same seed to cluster together
    valid_nclusts, silhouette_scores, DB_scores = [], [], []
    for n in range(min_n_clusts, max_n_clusts+1):
        clusters = fcluster(linkage_matrix, n, criterion="maxclust")
        if _check_seed_separation(clusters, n_assemblies_cum):
            valid_nclusts.append(n)
            silhouette_scores.append(silhouette_score(dists, clusters))
            DB_scores.append(davies_bouldin_score(dists, clusters))
    if len(valid_nclusts):
        assert n_method in ["min", "ss", "DB"], "Only silhouette scores and Davies-Bouldin index are supported atm."
        if n_method == "min":
            n_clust = valid_nclusts[0]
        elif n_method == "ss":
            n_clust = valid_nclusts[np.argmax(silhouette_scores)]
        elif n_method == "DB":
            n_clust = valid_nclusts[np.argmin(DB_scores)]
    else:
        raise RuntimeError("None of the cluster numbers in [%i, %i] fulfill the seed separation criteria"
                           % (min_n_clusts, max_n_clusts))
    clusters = fcluster(linkage_matrix, n_clust, criterion="maxclust")
    silhouettes = silhouette_samples(dists, clusters) if n_method == "ss" else None

    plotting = [linkage_matrix, silhouettes]
    return sim_matrix, clusters - 1, plotting


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


def cluster_synapses(c, post_gids, assembly, target_range, min_nsyns, log_sign_th=5.0, fig_dir=None):
    """
    Finds `min_nsyns` sized clusters of synapses within `target_range` (um) and tests their significance
    against a Poisson model (see `distance_model()` above) on all `post_gids` from the `assembly` passed
    :param c: bleupy circuit object
    :param post_gids: selected gids from the assembly in which the function will look for synapse clusters
    :param assembly: assembly object (used only for .gids and .idx)
    :param target_range: max distance from synapse center to consider for clustering
    :param min_nsyns: minimum number of synapses within one cluster
    :param log_sign_th: significance threshold (after taking log of p-values)
    :param fig_dir: optional debugging - if a proper dir. name is passed figures will be saved there
    :return: cluster_df: pandas DataFrame with one row per synapse and one column per 'label'
                         (for labels see `_create_lookups()` above)
                         placeholder: -100, synapse belonging to the label: -1, and synapse cluster idx start at 0
    """
    import pandas as pd
    import assemblyfire.utils as utils
    # as a circuit is passed `utils.get_bluepy_circuit()` would have thrown an error if bluepy was not installed...
    from bluepy.enums import Synapse
    xyz = [Synapse.POST_X_CENTER, Synapse.POST_Y_CENTER, Synapse.POST_Z_CENTER]
    syn_properties = [Synapse.PRE_GID, Synapse.POST_GID, Synapse.POST_SECTION_ID] + xyz
    if fig_dir is not None:
        utils.ensure_dir(fig_dir)

    gids = utils.get_E_gids(c, "hex_O1")  # TODO: not hard code target
    # get all necessary synapse properties in one go (and chunk later in the loop)
    syn_df = utils.get_syn_properties(c, utils.get_syn_idx(c, gids, post_gids), syn_properties)
    cluster_dfs = []
    for gid in post_gids:
        syn_df_gid = syn_df.loc[syn_df[Synapse.POST_GID] == gid]
        syn_idx_dict, fracs = _create_lookups(syn_df_gid, assembly)
        dists = syn_distances(syn_df_gid, Synapse.POST_SECTION_ID, xyz)
        if fig_dir is not None:
            fig_name = os.path.join(fig_dir, "assembly%i_a%i_synapse_dists.png" % (assembly.idx[0], gid))
            models = distance_model(deepcopy(dists), fracs, target_range, fig_name=fig_name)
        else:
            models = distance_model(deepcopy(dists), fracs, target_range)
        labels = list(syn_idx_dict.keys())
        results = -100 * np.ones((dists.shape[0], len(labels)), dtype=int)
        for i, label in enumerate(labels):
            syn_idx = syn_idx_dict[label]
            results[syn_idx, i] = -1
            sub_dists = dists[np.ix_(syn_idx, syn_idx)]
            nsyns = (sub_dists < target_range).sum(axis=1)
            p_vals = 1.0 - models[label].cdf(nsyns - 1)
            p_vals[p_vals == 0.0] += 1 / np.power(10, 2*log_sign_th)  # for numerical stability (see log10 below)
            significant = (-np.log10(p_vals) >= log_sign_th) & (nsyns >= min_nsyns)
            if np.any(significant):
                raw_clusters = sub_dists[:, significant] < target_range
                merged_clusters = merge_clusters(raw_clusters)
                row_idx, col_idx = np.nonzero(merged_clusters)
                results[syn_idx[row_idx], i] = col_idx  # set cluster labels (starting at 0)
        data = np.concatenate((syn_df_gid[[Synapse.PRE_GID, Synapse.POST_GID]].to_numpy(), results), axis=1)
        cluster_df = pd.DataFrame(data=data, index=syn_df_gid.index, columns=["pre_gid", "post_gid"] + labels)
        if fig_dir is not None:
            from assemblyfire.plots import plot_synapse_clusters
            morph = c.morph.get(int(gid), transform=True)
            fig_name = os.path.join(fig_dir, "assembly%i_a%i_synapse_clusters.png" % (assembly.idx[0], gid))
            plot_synapse_clusters(morph, pd.concat((cluster_df[labels], syn_df_gid[xyz]), axis=1), xyz, fig_name)
        cluster_dfs.append(cluster_df)
    return pd.concat(cluster_dfs).sort_index(inplace=True)
