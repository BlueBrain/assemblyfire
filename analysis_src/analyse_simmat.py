"""
Loads saved stuff, recalculates similarity matrix and analyse the temporal evolution of the activity
last modified: András Ecker 02.2022
"""

import os
import numpy as np
from scipy.spatial.distance import pdist, squareform

from assemblyfire.utils import load_spikes_from_h5
from assemblyfire.config import Config
from assemblyfire.clustering import cosine_similarity
from assemblyfire.plots import plot_mean_sims


def get_mats(h5f_name):
    """Reloads significant time bins and spike matrices from the HDF5 file and recalculates similarity matrices
    and also calculates the distance matrix between time bins (return upper triangular matrices)"""
    spike_matrix_dict, _ = load_spikes_from_h5(h5f_name)
    sim_matrix_dict, t_dists_dict = {}, {}
    for seed, spikes in spike_matrix_dict.items():
        t_dists_dict[seed] = pdist(spike_matrix_dict[seed].t_bins.reshape(-1, 1))
        sim_mat = cosine_similarity(spike_matrix_dict[seed].spike_matrix.T)
        np.fill_diagonal(sim_mat, 0.)  # stupid numpy...
        sim_matrix_dict[seed] = squareform(sim_mat)
    return t_dists_dict, sim_matrix_dict


def get_mean_sims_in_twindows(sim_mat, t_diffs, window_width=5000, window_shift=500):
    """Gets mean similarity within sliding time window as well as increasing time window (with fixed start)"""
    t_starts = np.arange(0, np.max(t_diffs) - window_width + window_shift, window_shift)
    mean_sims_sliding = np.array([np.mean(sim_mat[(t_start < t_diffs) & (t_diffs < t_start + window_width)])
                                  for t_start in t_starts])
    window_widths = t_starts[t_starts >= window_width]
    mean_sims_incr = np.array([np.mean(sim_mat[t_diffs < window_width]) for window_width in window_widths])
    return t_starts, mean_sims_sliding, window_widths, mean_sims_incr


def main(config_path):
    config = Config(config_path)
    ts, sim_mats = get_mats(config.h5f_name)
    for seed, sim_mat in sim_mats.items():
        t_starts, mean_sims_sliding, window_widths, mean_sims_incr = get_mean_sims_in_twindows(sim_mat, ts[seed])
        fig_name = os.path.join(config.fig_path, "mean_similarity_seed%i.png" % seed)
        plot_mean_sims(t_starts, mean_sims_sliding, window_widths, mean_sims_incr, fig_name)


if __name__ == "__main__":
    config_path = "../configs/v7_0-3mins.yaml"
    main(config_path)

