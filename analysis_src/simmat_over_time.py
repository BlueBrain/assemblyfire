"""
Loads saved stuff, recalculates similarity matrix and analyse the temporal evolution of the activity
last modified: András Ecker 02.2023
"""

import os
import numpy as np
from scipy.spatial.distance import pdist, squareform

from assemblyfire.config import Config
from assemblyfire.utils import load_spikes_from_h5
from assemblyfire.clustering import cosine_similarity
from assemblyfire.plots import plot_sim_vs_tdiff, plot_sim_vs_rate


def _pairwise_mean(x):
    """Vectorized pairwise mean (returns the same upper triangular format as `pdist`)"""
    i, j = np.triu_indices(len(x), k=1)
    return (x[i] + x[j]) / 2


def similarity_vs_rate_and_tdiff(spike_matrix, t_bins, window_width=5000, window_shift=500):
    """Gets (pairwise) similarity vs. pairwise average firing rate and
    mean similarity (within time window) against increasing temporal offset"""
    # get similarity and temporal distance "matrices"
    sim_matrix = cosine_similarity(spike_matrix.T)
    np.fill_diagonal(sim_matrix, 0.)  # stupid numpy...
    sim_matrix = squareform(sim_matrix)  # convert to upper triangular matrix
    t_dists = pdist(t_bins.reshape(-1, 1))
    # get pairwise mean rate (in the same format as the above 2)
    rate = np.sum(spike_matrix, axis=0)
    rate /= spike_matrix.shape[0] * 1e-3 * np.min(t_dists)  # last part should be config.bin_size...
    pw_avg_rate = _pairwise_mean(rate)
    # calculate mean similarity
    t_offsets = np.arange(window_shift, np.max(t_dists) - window_width + window_shift, window_shift)
    mean_sims = np.array([np.mean(sim_matrix[(t_offset < t_dists) & (t_dists < t_offset + window_width)])
                          for t_offset in t_offsets])
    return sim_matrix, pw_avg_rate, t_offsets, mean_sims


def main(config_path):
    config = Config(config_path)
    spike_matrix_dict, _ = load_spikes_from_h5(config.h5f_name)

    for seed, SpikeMatrixResult in spike_matrix_dict.items():
        spike_matrix, t_bins = SpikeMatrixResult.spike_matrix, SpikeMatrixResult.t_bins
        sim_matrix, pw_avg_rate, t_offsets, mean_sims = similarity_vs_rate_and_tdiff(spike_matrix, t_bins)
        plot_sim_vs_tdiff(t_offsets.copy(), mean_sims, os.path.join(config.fig_path, "similarity_vs_tdiff_%s.png" % seed))
        plot_sim_vs_rate(pw_avg_rate, sim_matrix, os.path.join(config.fig_path, "similarity_vs_rate_%s.png" % seed))


if __name__ == "__main__":
    main("../configs/v7_plastic.yaml")

