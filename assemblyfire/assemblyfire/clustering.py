# -*- coding: utf-8 -*-
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
last modified: András Ecker 02.2021
"""

import os
import logging
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
from scipy.optimize import curve_fit
from scipy.stats.distributions import t
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.cluster.hierarchy import ward, fcluster
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score

L = logging.getLogger("assemblyfire")


def cosine_similarity(X):
    """Cosine similarity matrix calculation (using only core numpy)
    much faster then `1-squareform(pdist(X, metrix="cosine"))`"""
    X_norm = X / np.linalg.norm(X, axis=-1)[:, np.newaxis]
    return np.dot(X_norm, X_norm.T)


# hierarchical clustering (using scipy and sklearn)
def cluster_sim_mat(spike_matrix, min_n_clusts=4, max_n_clusts=20):
    """Hieararchical (Ward linkage) clustering of cosine similarity matrix of significant time bins"""

    # cond_dists = pdist(spike_matrix.T, metric="cosine")
    # dists = squareform(cond_dists)
    # sim_matrix = 1 - dists

    sim_matrix = cosine_similarity(spike_matrix.T)
    dists = 1 - sim_matrix
    dists[dists < 1e-10] = 0.  # fixing numerical errors
    cond_dists = squareform(dists)  # squareform implements its inverse if the input is a square matrix

    linkage = ward(cond_dists)

    # determine number of clusters using silhouette scores (?Davies-Bouldin index?)
    silhouette_scores = []
    DB_scores = []
    for n in range(min_n_clusts, max_n_clusts+1):
        clusters = fcluster(linkage, n, criterion="maxclust")
        silhouette_scores.append(silhouette_score(dists, clusters))
        DB_scores.append(davies_bouldin_score(dists, clusters))
    n_clust = np.argmax(silhouette_scores) + min_n_clusts
    #n_clust = np.argmin(DB_scores) + min_n_clusts

    clusters = fcluster(linkage, int(n_clust), criterion="maxclust")
    silhouettes = silhouette_samples(dists, clusters)
    clusters = clusters - 1  # to start indexing at 0

    plotting = [linkage, silhouettes]
    return sim_matrix, clusters, plotting


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
    assert len(centroid_idx) <= 20, "len(centroid_idx)=%i > 20" % les(centroid_idx)

    plotting = [rhos, deltas, tmp_idx, sorted_gammas, fit, lCI, uCI, centroid_idx]

    # assign points to centroids
    clusters = np.argmin(dists[:, centroid_idx], axis=1)
    clusters[centroid_idx] = np.arange(len(centroid_idx))

    return clusters, plotting


# core-cells and assemblies related functions (mostly calculating correlations)
def pairwise_correlation_X(x):
    """Pairwise correlation between rows of a matrix x"""
    # sklearn.metrics.pairwise_distances can be parallelized (to be *much* faster),
    # but it doesn't return a symmetric correlation matrix ...
    corrs = 1 - squareform(pdist(x, metric="correlation"))
    corrs[np.isnan(corrs)] = 0.
    return corrs


def pairwise_correlation_XY(x, y):
    """Pairwise correlation between rows of a matrix X and matrix Y"""
    corrs = 1 - cdist(x, y, metric="correlation")
    corrs[np.isnan(corrs)] = 0.
    return corrs


def _convert_clusters(clusters):
    """Convert clusters into a sparse matrix form for `cdist()`"""
    sparse_clusters = np.zeros((len(np.unique(clusters)), clusters.shape[0]), dtype=np.int)
    for i in np.unique(clusters):
        sparse_clusters[i, clusters == i] = 1
    return sparse_clusters


def corr_spike_matrix_clusters(spike_matrix, sparse_clusters):
    """Correlation of cells with clusters (of time bins)"""
    return pairwise_correlation_XY(spike_matrix, sparse_clusters)


def corr_shuffled_spike_matrix_clusters(spike_matrix, sparse_clusters):
    """Random shuffles of each row of the spike matrix independently
    (keeps number of spikes per neuron) in order to create surrogate dataset
    for significance test of correlation"""

    for gid in range(spike_matrix.shape[0]):
        np.random.shuffle(spike_matrix[gid])
    return corr_spike_matrix_clusters(spike_matrix, sparse_clusters)


def _corr_spikes_clusters_subprocess(inputs):
    """Subprocess used by multiprocessing pool for setting significance threshold"""
    return corr_shuffled_spike_matrix_clusters(*inputs)


def sign_corr_ths(spike_matrix, sparse_clusters, N=1000):
    """Generates surrogate datasets and calculates correlation coefficients
    then takes 95% percentile of the surrogate datasets as a significance threshold"""

    n = N if mp.cpu_count()-1 > N else mp.cpu_count()-1
    pool = mp.Pool(processes=n)
    corrs = pool.map(_corr_spikes_clusters_subprocess, zip([deepcopy(spike_matrix) for _ in range(N)],
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


def _update_block_diagonal_dists(dists, n_assemblies):
    """
    Assemblies from the same seed tend to cluster together, but that's not what we want. - Daniela
    Thus, this function fills block diagonals with "infinite" distance representing infinite distance
    between different assemblies from the same seed and return scipy's condensed distance representation
    which can be passed to hierarhichal clustering in the next step
    """

    inf_dist = np.max(dists) * 10
    n_assemblies_cum = [0] + np.cumsum(n_assemblies).tolist()
    for i, j in zip(n_assemblies_cum[:-1], n_assemblies_cum[1:]):
        dists[i:j, i:j] = inf_dist
    np.fill_diagonal(dists, 0)
    return squareform(dists)


def cluster_spikes(spike_matrix_dict, method, FigureArgs):
    """
    Cluster spikes either via hierarchical clustering (Ward's linkage)
    of the cosine similarity matrix of significant time bins (see Perez-Ortega et al. 2020), or
    density based clustering of spike matrix projected to PCA space (see Herzog et al. 2020)
    :param spike_matrix_dict: dict with seed as key and SpikeMatrixResult (see `spikes.py`) as value
    :param method: str - clustering method (read from yaml config file)
    :param FigureArgs: plotting related arguments (see `cli.py`)
    :return: dict with seed as key and clustered (significant) time bins as value
    """
    from assemblyfire.plots import plot_cluster_seqs, plot_pattern_clusters

    clusters_dict = {}
    for seed, SpikeMatrixResult in tqdm(spike_matrix_dict.items(), desc="Clustering"):
        spike_matrix = SpikeMatrixResult.spike_matrix
        t_bins = SpikeMatrixResult.t_bins

        if method == "hierarchical":
            from assemblyfire.plots import plot_sim_matrix, plot_dendogram_silhouettes
            sim_matrix, clusters, plotting = cluster_sim_mat(spike_matrix)

            fig_name = os.path.join(FigureArgs.fig_path, "similarity_matrix_seed%i.png" % seed)
            plot_sim_matrix(sim_matrix, t_bins, FigureArgs.stim_times, FigureArgs.patterns, fig_name)
            fig_name = os.path.join(FigureArgs.fig_path, "Ward_clustering_seed%i.png" % seed)
            plot_dendogram_silhouettes(clusters, *plotting, fig_name)

        elif method == "density_based":
            from assemblyfire.plots import plot_transformed, plot_components, plot_rhos_deltas
            pca_transformed, pca_components = PCA_ncomps(spike_matrix.T, 12)
            clusters, plotting = db_clustering(pca_transformed)

            fig_name = os.path.join(FigureArgs.fig_path, "PCA_transformed_seed%i.png" % seed)
            plot_transformed(pca_transformed, t_bins, FigureArgs.stim_times, FigureArgs.patterns, fig_name)
            fig_name = os.path.join(FigureArgs.fig_path, "PCA_components_seed%i.png" % seed)
            plot_components(pca_components, SpikeMatrixResult.gids, FigureArgs.depths, fig_name)
            fig_name = os.path.join(FigureArgs.fig_path, "rho_delta_seed%i.png" % seed)
            plot_rhos_deltas(*plotting, fig_name)

        clusters_dict[seed] = clusters
        fig_name = os.path.join(FigureArgs.fig_path, "cluster_seq_seed%i.png" % seed)
        plot_cluster_seqs(clusters, t_bins, FigureArgs.stim_times, FigureArgs.patterns, fig_name)
        fig_name = os.path.join(FigureArgs.fig_path, "clusters_patterns_seed%i.png" % seed)
        plot_pattern_clusters(clusters, t_bins, FigureArgs.stim_times, FigureArgs.patterns, fig_name)

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
        core_cell_idx = np.zeros_like(corrs, dtype=np.int)
        core_cell_idx[np.where(corrs > corr_ths)] = 1
        # cell assemblies
        assembly_idx = within_cluster_correlations(spike_matrix, core_cell_idx)

        # save to h5
        metadata = {"clusters": clusters}
        assembly_lst = [Assembly(gids[core_cell_idx[:, i] == 1], index=(i, seed))
                                 for i in assembly_idx]
        assemblies = AssemblyGroup(assemblies=assembly_lst, all_gids=gids,
                                   label="seed%i" % seed, metadata=metadata)
        assemblies.to_h5(h5f_name, prefix=h5_prefix)

        # plot (only depth profile at this point)
        fig_name = os.path.join(FigureArgs.fig_path, "assemblies_seed%i.png" % seed)
        plot_assemblies(core_cell_idx, assembly_idx, gids,
                        FigureArgs.ystuff, FigureArgs.depths, fig_name)


def cluster_assemblies(assemblies, n_assemblies, criterion, criterion_arg):
    """
    Hieararchical (Ward linkage) clustering of hamming similarity matrix of assemblies from different seeds
    :param assemblies: assemblies x gids boolean array representing all assemblies across seeds
    :param n_assemblies: list with number of assemblies per seed
    :param criterion: criterion for hierarchical clustering
    :param criterion_arg: if criterion is maxclust the number of clusters to find
                          if criterion is distance the threshold to cut the dendogram
    """

    cond_dists = pdist(assemblies, metric="hamming")
    dists = squareform(cond_dists)
    sim_matrix = 1 - dists

    cond_dists = _update_block_diagonal_dists(dists, n_assemblies)

    linkage = ward(cond_dists)

    clusters = fcluster(linkage, criterion_arg, criterion=criterion)
    silhouettes = silhouette_samples(dists, clusters)
    clusters = clusters - 1  # to start indexing at 0
    cluster_idx, counts = np.unique(clusters, return_counts=True)
    L.info("Total number of assemblies: %i, Number of clusters: %i" % (np.sum(counts), len(counts)))

    plotting = [linkage, silhouettes]
    return sim_matrix, clusters, plotting