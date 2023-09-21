"""
Functions to run hierarchical clustering (via Ward's linkage)
of the cosine similarity matrix of significant time bins
a la Perez-Ortega et al. 2021 (see also Carillo-Reid et al. 2015)
Then "core-cells" and cell assemblies are detected with correlation
based methods from (Montijn et al. 2016 and) Herzog et al. 2021.
Assemblies are clustered into consensus assemblies via hierarchical clustering.
Also implements methods to cluster synapses based on their location on the dendrites
authors: András Ecker, Michael W. Reimann; last modified: 07.2023
"""

import os
import logging
from tqdm import tqdm
from hashlib import md5
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import ttest_ind, poisson
from scipy.sparse import csr_matrix, issparse
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score

L = logging.getLogger("assemblyfire")
XYZ = ["x", "y", "z"]
DSET_CLST = "strength"
DSET_MEMBER = "member"
DSET_PVALUE = "pvalue"


def cosine_similarity(x):
    """Cosine similarity between rows of matrix
    much faster than `1 - squareform(pdist(x, metrix="cosine"))`"""
    x_norm = x / np.linalg.norm(x, axis=-1)[:, np.newaxis]
    return np.dot(x_norm, x_norm.T)


# hierarchical clustering (using scipy and sklearn)
def cluster_sim_mat(spike_matrix, min_n_clusts=5, max_n_clusts=20, n_method="DB"):
    """Hieararchical (Ward linkage) clustering of cosine similarity matrix of significant time bins"""

    sim_matrix = cosine_similarity(spike_matrix.T)
    dists = 1 - sim_matrix
    dists[dists < 1e-5] = 0.  # fixing numerical errors
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


# core-cells and assemblies related functions (mostly calculating correlations)
def pairwise_correlation_x(sparse_x):
    """Pairwise correlation between rows of a matrix `x` (assumed to be sparse)
    much faster than `1 - squareform(pdist(x, metrix="correlation"))`"""
    assert issparse(sparse_x), "Matrix is expected to be in sparse format"
    n = sparse_x.shape[1]
    row_sum = sparse_x.sum(axis=1, dtype=np.float32)
    cov = (sparse_x * sparse_x.T - (np.outer(row_sum, row_sum) / n)) / (n - 1)
    norm = np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    return cov / (norm + 1e-40)


def pairwise_correlation_xy(x, y):
    """Pairwise correlation between rows of a matrix X and matrix Y"""
    corrs = 1 - cdist(x, y, metric="correlation")
    corrs[np.isnan(corrs)] = 0.
    return corrs


def _convert_clusters(clusters):
    """Convert cluster vector into a matrix form for `cdist()`"""
    sparse_clusters = np.zeros((len(np.unique(clusters)), clusters.shape[0]), dtype=int)
    for i in np.unique(clusters):
        sparse_clusters[i, clusters == i] = 1
    return sparse_clusters


def corr_spike_matrix_clusters(spike_matrix, clusters):
    """Correlation of cells with clusters (of time bins)"""
    return pairwise_correlation_xy(spike_matrix, clusters)


def corr_shuffled_spike_matrix_clusters(spike_matrix, clusters):
    """Random shuffles columns of the spike matrix (keeps number of spikes per neuron)
    in order to create surrogate dataset for significance test for correlation"""
    spike_matrix_rnd = spike_matrix[:, np.random.permutation(spike_matrix.shape[1])]
    del spike_matrix
    return corr_spike_matrix_clusters(spike_matrix_rnd, clusters)


def sign_corr_ths(spike_matrix, clusters, th_pct, nreps=1000):
    """Generates `N` surrogate datasets and calculates correlation coefficients
    then takes `th_pct`% percentile of the surrogate datasets as a significance threshold"""
    nprocs = nreps if os.cpu_count()-1 > nreps else os.cpu_count()-1
    nprocs = 20 if nprocs > 20 and spike_matrix.shape[1] > 5000 else nprocs  # limit processes to not run out of memory
    with Parallel(n_jobs=nprocs, prefer="threads") as p:
        corrs = p(delayed(corr_shuffled_spike_matrix_clusters)(spike_matrix, clusters) for _ in range(nreps))
    corrs = np.dstack(corrs)  # shape: ngids x nclusters x N
    # get sign threshold (compare to Monte-Carlo shuffles)
    corr_ths = np.percentile(corrs, th_pct, axis=2, overwrite_input=True)
    return corr_ths


def get_core_cell_idx(spike_matrix, clusters, th_pct):
    """Finds cells whose spiking activity correlate with the activation of (clustered) time bins
    (Just a combinations of functions above, made into yet another function to be callable for other scripts...)"""
    assert spike_matrix.shape[1] == len(clusters)
    clusters = _convert_clusters(clusters)
    corrs = corr_spike_matrix_clusters(spike_matrix, clusters)
    corr_ths = sign_corr_ths(spike_matrix, clusters, th_pct)
    core_cell_idx = np.zeros_like(corrs, dtype=int)
    core_cell_idx[corrs > corr_ths] = 1
    return core_cell_idx, corrs


def within_cluster_correlations(spike_matrix, core_cell_idx):
    """Compares within cluster correlations (correlation of core cells)
    against the avg. correlation in the whole dataset
    if the within cluster correlation it's higher the cluster is an assembly"""
    corrs = pairwise_correlation_x(spike_matrix)
    np.fill_diagonal(corrs, np.nan)
    mean_corr = np.nanmean(corrs)

    assembly_idx = []
    for i in range(core_cell_idx.shape[1]):
        idx = np.where(core_cell_idx[:, i] == 1)[0]
        if np.nanmean(corrs[np.ix_(idx, idx)]) > mean_corr:
            assembly_idx.append(i)
    return assembly_idx


def cluster_spikes(spike_matrix_dict, overwrite_seeds, project_metadata, fig_path):
    """
    Cluster spikes either via hierarchical clustering (Ward's linkage)
    of the cosine similarity matrix of significant time bins (see Perez-Ortega et al. 2021), or
    density based clustering of spike matrix projected to PCA space (see Herzog et al. 2021)
    :param spike_matrix_dict: dict with seed as key and SpikeMatrixResult (see `spikes.py`) as value
    :param overwrite_seeds: dict with seeds as keys and values as desired number of clusters
                           (instead of the optimal cluster number determined with the current heuristics in place)
    :param FigureArgs: plotting related arguments (see `find_assemblies.py`)
    :return: dict with seed as key and clustered (significant) time bins as value
    """
    from assemblyfire.plots import plot_sim_matrix, plot_cluster_seqs, plot_pattern_clusters, plot_dendogram_silhouettes

    ts, stim_times = project_metadata["t"], project_metadata["stim_times"]
    patterns = np.array(project_metadata["patterns"])
    clusters_dict = {}
    for seed, SpikeMatrixResult in tqdm(spike_matrix_dict.items(), desc="Clustering"):
        spike_matrix, t_bins = SpikeMatrixResult.spike_matrix, SpikeMatrixResult.t_bins
        if len(ts) > 2:  # if t is chunked the chunk bounds are saved and "seeds" represent chunks
            idx = np.where((ts[seed] <= stim_times) & (stim_times < ts[seed + 1]))[0]
        else:
            idx = np.arange(len(stim_times))  # just to not break the code...

        if "seed%s" % seed not in overwrite_seeds:
            sim_matrix, clusters, plotting = cluster_sim_mat(spike_matrix)
        else:
            sim_matrix, clusters, plotting = cluster_sim_mat(spike_matrix, min_n_clusts=overwrite_seeds["seed%s" % seed],
                                                             max_n_clusts=overwrite_seeds["seed%s" % seed])
        clusters_dict[seed] = clusters

        fig_name = os.path.join(fig_path, "similarity_matrix_seed%s.png" % seed)
        plot_sim_matrix(sim_matrix.copy(), t_bins, stim_times[idx], patterns[idx], fig_name)
        fig_name = os.path.join(fig_path, "ward_clustering_seed%s.png" % seed)
        plot_dendogram_silhouettes(clusters, *plotting, fig_name)
        fig_name = os.path.join(fig_path, "cluster_seq_seed%s.png" % seed)
        plot_cluster_seqs(clusters, t_bins, stim_times[idx], patterns[idx], fig_name)
        fig_name = os.path.join(fig_path, "clusters_patterns_seed%s.png" % seed)
        plot_pattern_clusters(clusters, t_bins, stim_times[idx], patterns[idx], fig_name)
    return clusters_dict


def detect_assemblies(spike_matrix_dict, clusters_dict, core_cell_th_pct, h5f_name, h5_prefix, nrn_loc_df, fig_path):
    """
    Finds "core cells" - cells which correlate with the activation of (clustered) time bins
    and then checks within group correlations against the mean correlation to decide
    if the core cell group is an assembly or not a la Herzog et al. 2021
    :param spike_matrix_dict: dict with seed as key and SpikeMatrixResult (see `spikes.py`) as value
    :param clusters_dict: dict with seed as key and clustered (significant) time bins as value (see `cluster_spikes()`)
    :param core_cell_th_pct: float - sign. threshold in surrogate dataset for core cell detection
    :param h5f_name: str - name of the HDF5 file (dumping the assemblies and their metadata)
    :param h5_prefix: str - directory name of assemblies within the HDF5 file
    :param nrn_loc_df: DataFrame with neuron locations or None to skip plotting
    :param fig_path: str - root path for figures
    """
    from assemblyfire.assemblies import Assembly, AssemblyGroup
    from assemblyfire.plots import plot_assemblies

    for seed, SpikeMatrixResult in tqdm(spike_matrix_dict.items(), desc="Detecting assemblies"):
        spike_matrix = SpikeMatrixResult.spike_matrix
        gids = SpikeMatrixResult.gids
        clusters = clusters_dict[seed]

        core_cell_idx, _ = get_core_cell_idx(spike_matrix, clusters, core_cell_th_pct)
        spike_matrix_csr = csr_matrix(spike_matrix, dtype=np.float32)
        del spike_matrix
        assembly_idx = within_cluster_correlations(spike_matrix_csr, core_cell_idx)

        # save to h5
        metadata = {"clusters": clusters}
        assembly_lst = []
        for i in assembly_idx:
            index = (i, seed) if seed != "_average" else i  # "`_average` cannot be saved as h5 attr."
            assembly_lst.append(Assembly(gids[core_cell_idx[:, i] == 1], index=index))
        assemblies = AssemblyGroup(assemblies=assembly_lst, all_gids=gids, label="seed%s" % seed, metadata=metadata)
        assemblies.to_h5(h5f_name, prefix=h5_prefix)
        # plot (only spatial location at this point)
        if nrn_loc_df is not None:
            fig_name = os.path.join(fig_path, "assemblies_seed%s.png" % seed)
            plot_assemblies(core_cell_idx, assembly_idx, gids, nrn_loc_df, fig_name)


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
    inf_dist = np.max(dists) * 2
    for i, j in zip(n_assemblies_cum[:-1], n_assemblies_cum[1:]):
        dists[i:j, i:j] = inf_dist
    np.fill_diagonal(dists, 0)
    return squareform(dists)


def cluster_assemblies(assemblies, n_assemblies, distance_metric, linkage_method,
                       update_block_diagonals=True, n_method="min"):
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


def syn_nearest_neighbour_distances(gid, mpdc, syn_loc_df, assembly_grp, same_section_only=False, n_ctrls=20):
    """
    Calculate nearest neighbour distance for all synaptic locations along the dendrite.
    (+ Compares it to a random controls, with the same number of presynaptic gids)
    param gid (int): gid of the neuron whose dendritic locations are considered
    param mpdc (conntility.subcellular.MorphologyPathDistanceCalculator): Path distance calculator for the same neuron
    param syn_loc_df (pd.DataFrame): frame specifying dendritic locations on the dendrite. (one row per loc)
                                     Columns are: ["afferent_section_id", "afferent_segment_id", "afferent_segment_offset"].
                                     (Indexed by the (presynaptic) gids associated with the locations.)
    param assembly_grp (AssemblyGroup): group of assemblies that are to be tested for clustering.
                                        The test is whether the dendritic locations associated with the assembly.gids
                                        are closer together than the location associated with an equally sized group.
    param same_section_only (bool, default=False): If True, only locations on the same dendritic section are considered.
    param n_ctrls (int): number of controls to use for t-test
    """
    results = {}
    for assembly in assembly_grp:
        results[("gid", "gid")] = gid
        results[("assembly%i" % assembly.idx[0], DSET_MEMBER)] = gid in assembly.gids

        from_assembly = np.in1d(syn_loc_df.index.to_numpy(), assembly.gids)
        from_assembly_count = len(np.intersect1d(syn_loc_df.index.to_numpy(), assembly.gids))
        if from_assembly.sum() == 0:
            results[("assembly%i" % assembly.idx[0], DSET_CLST)]: np.NaN
            results[("assembly%i" % assembly.idx[0], DSET_PVALUE)]: np.NaN
            continue
        pd_data = mpdc.path_distances(syn_loc_df[from_assembly], same_section_only=same_section_only)
        pd_data[pd_data == 0.] = np.NaN  # don't use distance to itself...
        nnd_data = np.nanmin(pd_data, axis=0)

        nnd_ctrl = []
        gids = np.unique(syn_loc_df.index.to_numpy())
        hash_ = md5(assembly.gids)
        assembly_seed = np.mod(int(hash_.hexdigest(), 16), 1000)
        for seed in range(n_ctrls):
            np.random.seed(seed * (assembly_seed + gid))
            from_ctrl = np.in1d(syn_loc_df.index.to_numpy(), np.random.choice(gids, from_assembly_count, replace=False))
            pd_ctrl = mpdc.path_distances(syn_loc_df[from_ctrl], same_section_only=same_section_only)
            pd_ctrl[pd_ctrl == 0] = np.NaN
            nnd_ctrl.append(np.nanmin(pd_ctrl, axis=0))

        a = np.mean(nnd_data)
        b = [np.mean(_ctrl) for _ctrl in nnd_ctrl]
        str_res = (a - np.nanmean(b)) / np.nanstd(b)
        stat_res = ttest_ind(nnd_data, np.hstack(nnd_ctrl), nan_policy="omit")
        results[("assembly%i" % assembly.idx[0], DSET_CLST)] = -str_res  # -1 to convert low nnd. to "high strength"
        results[("assembly%i" % assembly.idx[0], DSET_PVALUE)] = stat_res.pvalue

    return results


def _create_lookups(loc_df, assembly_grp):
    """Create dicts with synapse idx, and fraction of those (compared to total) for all assemblies
    in the `assembly_grp`. (As neurons can be part of more than 1 assembly, `fracs` won't add up to 1)"""
    syn_idx, fracs, all_assembly_syn_idx = {}, {}, []
    nsyns, all_pre_gids = len(loc_df), loc_df["pre_gid"].to_numpy()
    for assembly in assembly_grp:
        idx = np.in1d(all_pre_gids, assembly.gids)
        assembly_frac = idx.sum() / len(idx)
        fracs["assembly%i" % assembly.idx[0]] = assembly_frac
        assembly_idx = np.nonzero(idx)[0]
        syn_idx["assembly%i" % assembly.idx[0]] = assembly_idx
        all_assembly_syn_idx.append(assembly_idx)
    # finds synapses that aren't coming from any assembly
    all_assembly_syn_idx = np.unique(np.concatenate(all_assembly_syn_idx))  # neurons can be part of more than 1 assemblies...
    tmp = np.arange(nsyns)
    non_assembly_syn_idx = tmp[np.in1d(tmp, all_assembly_syn_idx, assume_unique=True, invert=True)]
    assert (len(all_assembly_syn_idx) + len(non_assembly_syn_idx) == nsyns), "Synapse numbers don't add up..."
    syn_idx["non_assembly"] = non_assembly_syn_idx
    fracs["non_assembly"] = len(non_assembly_syn_idx) / nsyns
    return syn_idx, fracs


def syn_distances(loc_df, mask_col, xzy_cols):
    """Return (Euclidean) distance between synapses on the same section (the rest is masked with nans)"""
    dists = squareform(pdist(loc_df[xzy_cols].to_numpy()))
    mask = squareform(pdist(loc_df[mask_col].to_numpy().reshape(-1, 1)))
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
    and if there are more than a continuously decreasing threshold `nsyns_th` (most outer for loop),
    then adds those to cluster `i` and deletes cluster `j`.
    (This is needed as clusters can be detected multiple times at the first place.
    Input `cluster.shape[1]` is the number of raw clusters,
    while output `cluster.shape[1]` is the number of merged clusters)
    """
    row_idx, col_idx = np.nonzero(clusters)
    nsyns = len(np.unique(row_idx))  # number of unique synapses in the passed clusters
    _, counts = np.unique(col_idx, return_counts=True)
    min_nsyns = np.min(counts)  # minimum number of synapses in the passed clusters
    nclusts_ub = int(nsyns / min_nsyns)  # upper bound of possible (meaningfull) clusters
    for nsyns_th in np.arange(min_nsyns-1, 1, -1):
        i = 0
        while i < (clusters.shape[1] - 1):
            idx_partners = np.arange(i + 1, clusters.shape[1])
            matches = np.sum(clusters[:, [i]] & clusters[:, idx_partners], axis=0) > nsyns_th
            if not np.any(matches):
                i += 1
            else:
                # idx_partners[matches] are the column indices (`j`s) where cluster `i` has at least 1 common synapse
                clusters[:, i] = np.any(clusters[:, [i] + idx_partners[matches].tolist()], axis=1)
                clusters = clusters[:, ~np.in1d(range(clusters.shape[1]), idx_partners[matches])]
                i += 1
        row_idx, _ = np.nonzero(clusters)
        _, counts = np.unique(row_idx, return_counts=True)
        if np.sum(counts) == nsyns:  # check if all synapses (in the rows) belong to a unique cluster
            break
    assert clusters.shape[1] <= nclusts_ub, "After merging there are still more clusters (%i)" \
                                            "than the theoretical upper bound: %i" % (clusters.shape[1], nclusts_ub)
    return clusters


def cluster_synapses(loc_df, assembly_grp, target_range, min_nsyns, log_sign_th=5.0,
                     fig_dir=None, base_assembly_idx=None):
    """
    Finds `min_nsyns` sized clusters of synapses within `target_range` (um) and tests their significance
    against a Poisson model (see `distance_model()` above) on all post_gids (passed in `loc_df`) from the assemblies
    passed in the `assembly_grp`
    :param loc_df: pandas DataFrame with synapse properties: pre-post gid, section id, x,y,z coordinates
    :param assembly_grp: AssemblyGroup object - synapse clusters (on `post_gids`) will be detected
                         from all assemblies passed (and for the remaining non-assembly neurons)
    :param target_range: max distance from synapse center to consider for clustering
    :param min_nsyns: minimum number of synapses within one cluster
    :param log_sign_th: significance threshold (after taking log of p-values)
    :param fig_dir: optional debugging - if a proper dir. name is passed figures will be saved there
    :param base_assembly_idx: ID of the base assembly in the group (if more than 1 is passed in `assembly_grp`)
                              (This is only used for naming figures if `fig_dir` is not None)
    :return: cluster_df: pandas DataFrame with one row per synapse and one column per 'label'
                         (for labels see `_create_lookups()` above)
                         placeholder: -100, synapse belonging to the label: -1, and synapse cluster idx start at 0
    """

    cluster_dfs = []
    for gid in loc_df["post_gid"].unique():
        loc_df_gid = loc_df.loc[loc_df["post_gid"] == gid]
        syn_idx_dict, fracs = _create_lookups(loc_df_gid, assembly_grp)
        dists = syn_distances(loc_df_gid, "section_id", XYZ)
        if fig_dir is not None:
            fig_name = os.path.join(fig_dir, "assembly%i_a%i_synapse_dists.png" % (base_assembly_idx, gid))
            models = distance_model(dists.copy(), fracs, target_range, fig_name=fig_name)
        else:
            models = distance_model(dists.copy(), fracs, target_range)
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
        data = np.concatenate((loc_df_gid[["pre_gid", "post_gid"]].to_numpy(), results), axis=1)
        cluster_df = pd.DataFrame(data=data, index=loc_df_gid.index, columns=["pre_gid", "post_gid"] + labels)
        cluster_dfs.append(cluster_df)
    return pd.concat(cluster_dfs)
