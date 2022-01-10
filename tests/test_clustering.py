"""
Testing `clustering.py/cluster_sim_mat()` with various nr. of clusters
last modified: András Ecker 01.2022
"""

import os
from copy import deepcopy
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_samples, davies_bouldin_score

from assemblyfire.config import Config
from assemblyfire.utils import load_spikes_from_h5
from assemblyfire.clustering import cosine_similarity
from assemblyfire.plots import plot_dendogram_silhouettes, plot_cluster_seqs


def _delete_seeds(spike_matrix_dict, keep_seeds):
    """Small helper to keep only a subset of seeds from the spike matrix dictionary"""
    spike_matrix_dict_keep_seeds = deepcopy(spike_matrix_dict)
    for seed, _ in spike_matrix_dict.items():
        if seed not in keep_seeds:
            del spike_matrix_dict_keep_seeds[seed]
    return spike_matrix_dict_keep_seeds


def test_cluster_sim_mat(spike_matrix_dict, stim_times, patterns, fig_dir):
    """Testing clustering.py/cluster_sim_mat() with various nr. of clusters"""
    for seed, SpikeMatrixResult in tqdm(spike_matrix_dict.items(), desc="Clustering"):
        spike_matrix = SpikeMatrixResult.spike_matrix
        t_bins = SpikeMatrixResult.t_bins
        sim_matrix = cosine_similarity(spike_matrix.T)
        dists = 1 - sim_matrix
        dists[dists < 1e-10] = 0.  # fixing numerical errors
        cond_dists = squareform(dists)  # squareform implements its inverse if the input is a square matrix
        linkage_matrix = linkage(cond_dists, method="ward")

        silhouette_scores, db_scores = [], []
        for n in range(5, 21):
            clusters = fcluster(linkage_matrix, n, criterion="maxclust")
            silhouettes = silhouette_samples(dists, clusters)
            silhouette_scores.append(np.mean(silhouettes))
            db_scores.append(davies_bouldin_score(dists, clusters))
            fig_name = os.path.join(fig_dir, "test_clustering", "Ward_clustering_seed%i_n%i.png" % (seed, n))
            plot_dendogram_silhouettes(clusters-1, linkage_matrix, silhouettes, fig_name)
            fig_name = os.path.join(fig_dir, "test_clustering", "cluster_seq_seed%i_n%i.png" % (seed, n))
            plot_cluster_seqs(clusters, t_bins, stim_times, patterns, fig_name)
        print("ss:", silhouette_scores)
        print("db:", db_scores)
        print(seed, "ss:", np.argmax(silhouette_scores)+5, "DB:", np.argmin(db_scores)+5)


if __name__ == "__main__":
    config = Config("../configs/v7_bbp-workflow.yaml")
    spike_matrix_dict, project_metadata = load_spikes_from_h5(config.h5f_name)
    # check only seed 100 from e0fbb0c8-07a4-49e0-be7d-822b2b2148fb
    spike_matrix_dict = _delete_seeds(spike_matrix_dict, [100])
    test_cluster_sim_mat(spike_matrix_dict, project_metadata["stim_times"],
                         project_metadata["patterns"], config.fig_path)


