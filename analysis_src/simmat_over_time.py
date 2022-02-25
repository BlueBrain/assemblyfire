"""
Loads saved stuff, recalculates similarity matrix and analyse the temporal evolution of the activity
last modified: András Ecker 02.2022
"""

import os
import numpy as np
from scipy.spatial.distance import pdist, squareform

from assemblyfire.config import Config
from assemblyfire.utils import load_spikes_from_h5, get_sim_path
from assemblyfire.spikes import load_spikes, spikes2mat
from assemblyfire.clustering import cosine_similarity
from assemblyfire.plots import plot_sims_vs_rate_and_tdiff


def generate_test_dataset(n_patterns=1000, stim_duration=500, late_assembly_offset=100):
    """Generate stereotypical toy dataset for testing the method"""
    # One randomly selected "assembly" followed by the late assembly
    rnd_seq = np.hstack([[np.random.randint(9) + 1, 0] for _ in range(n_patterns)])
    # Active instantly upon stim, late assembly a bit later
    ts = np.hstack([[t * stim_duration, t * stim_duration + late_assembly_offset] for t in range(n_patterns)])
    t_dists = pdist(ts.reshape(-1, 1))
    # similarity: 1 for matching assemblies, 0 for a mismatch
    sim_mat = np.vstack([rnd_seq == r for r in rnd_seq]).astype(int)
    np.fill_diagonal(sim_mat, 0.)  # stupid numpy...
    sim_mat = squareform(sim_mat)
    return t_dists, sim_mat


def get_thresholded_spikes(config, spike_matrix_dict):
    """Gets spikes from the whole long simulation and thresholds them
    based on the (saved) significance thresholds in time chunks.
    (Slower than loading chunk wise and merging would be, but easier to implement as only spiking gids are stored
    and those don't necessarily match within time chunks...)"""
    sim_paths = get_sim_path(config.root_path)
    spike_times, spiking_gids = load_spikes(sim_paths.iloc[0], config.target, config.t_start, config.t_end)
    spike_matrix, gids, t_bins = spikes2mat(spike_times, spiking_gids, config.t_start, config.t_end, config.bin_size)
    sign_t_bins = np.concatenate([spikes.t_bins for _, spikes in spike_matrix_dict.items()])
    spike_matrix = spike_matrix[:, np.in1d(t_bins[:-1], sign_t_bins, assume_unique=True)]
    return spike_matrix, gids, sign_t_bins


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
    t_offsets = np.arange(0, np.max(t_dists) - window_width + window_shift, window_shift)
    mean_sims = np.array([np.mean(sim_matrix[(t_offset < t_dists) & (t_dists < t_offset + window_width)])
                          for t_offset in t_offsets])
    return sim_matrix, pw_avg_rate, t_offsets, mean_sims


def main(config_path):
    config = Config(config_path)
    spike_matrix_dict, project_metadata = load_spikes_from_h5(config.h5f_name)
    if len(project_metadata["t"]) == 2:
        raise RuntimeError("This script only works for chunked simulations!")

    spike_matrix, _, t_bins = get_thresholded_spikes(config, spike_matrix_dict)
    sim_matrix, pw_avg_rate, t_offsets, mean_sims = similarity_vs_rate_and_tdiff(spike_matrix, t_bins)
    fig_name = os.path.join(config.fig_path, "similarity_vs_rate_and_time.png")
    plot_sims_vs_rate_and_tdiff(sim_matrix, pw_avg_rate, t_offsets, mean_sims, fig_name)


if __name__ == "__main__":
    config_path = "../configs/v7_10mins.yaml"
    main(config_path)

