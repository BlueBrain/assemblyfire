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
from assemblyfire.plots import plot_mean_sims


def generate_test_dataset(n_patterns=1000, stim_duration=500, late_assembly_offset=100):
    """Generate stereotypical toy dataset for testing the method"""
    # One randomly selected "assembly" instantly active, followed by the late assembly
    rnd_seq = np.hstack([[np.random.randint(9) + 1, 0] for _ in range(n_patterns)])
    # Active instantly upon stim; zero assembly a bit later
    ts = np.hstack([[t * stim_duration, t * stim_duration + late_assembly_offset] for t in range(n_patterns)])
    t_dists = pdist(ts.reshape(-1, 1))
    # similarity = 1 for matching assemblies; 0 for a mismatch
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


def get_mean_sims_vs_tdiff(spike_matrix, t_bins, window_width=5000, window_shift=500):
    """Gets mean similarity (within time window) against increasing temporal separation"""
    # get similarity and temporal distance matrices
    sim_matrix = cosine_similarity(spike_matrix.T)
    np.fill_diagonal(sim_matrix, 0.)  # stupid numpy...
    sim_matrix = squareform(sim_matrix)  # convert to upper triangular matrix
    t_dists = pdist(t_bins.reshape(-1, 1))
    # calculate mean similarity
    t_starts = np.arange(0, np.max(t_dists) - window_width + window_shift, window_shift)
    mean_sims = np.array([np.mean(sim_matrix[(t_start < t_dists) & (t_dists < t_start + window_width)])
                          for t_start in t_starts])
    return t_starts, mean_sims


def main(config_path):
    config = Config(config_path)
    spike_matrix_dict, project_metadata = load_spikes_from_h5(config.h5f_name)
    if len(project_metadata["t"]) == 2:
        raise RuntimeError("This script only works for chunked simulations!")

    spike_matrix, _, t_bins = get_thresholded_spikes(config, spike_matrix_dict)
    t_starts, mean_sims = get_mean_sims_vs_tdiff(spike_matrix, t_bins)
    fig_name = os.path.join(config.fig_path, "mean_similarity.png")
    plot_mean_sims(t_starts, mean_sims, fig_name)


if __name__ == "__main__":
    config_path = "../configs/v7_10mins.yaml"
    main(config_path)

