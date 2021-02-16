# -*- coding: utf-8 -*-
"""

last modified: András Ecker 02.2021
"""

import os
from collections import namedtuple
import numpy as np
from gen_test_spikes import parse_filename
from assemblyfire.spikes import spikes2mat, sign_rate_std
from assemblyfire.clustering import cluster_spikes
from assemblyfire.utils import get_patterns, get_stim_times
from assemblyfire.plots import plot_rate

SpikeMatrixResult = namedtuple("SpikeMatrixResult", ["spike_matrix", "gids", "t_bins"])
FigureArgs = namedtuple("FigureArgs", ["stim_times", "patterns", "depths", "ystuff", "fig_path"])
inp_dir = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/simulations/test_spiketrains"
fig_path = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/figures/test_spiketrains"


def load_spikes(f_name):
    """Loads generated spikes from out.dat"""
    spike_times = []
    spiking_gids = []
    with open(f_name, "r") as f:
        next(f)
        for l in f:
            tmp = l.strip().split()
            spike_times.append(float(tmp[0]))
            spiking_gids.append(int(tmp[1]))
    return np.asarray(spike_times), np.asarray(spiking_gids)


def get_sign_spike_matrices(patterns_fname, spikes_fname, seed, fig_path, bin_size=10):
    """Quick and dirty rewrite of `assemblyfire/spikes.py/SpikeMatrixGroup.get_sign_spike_matrices()` for testing"""

    spike_times, spiking_gids = load_spikes(spikes_fname)
    patterns = get_patterns(patterns_fname)
    stim_times = get_stim_times(patterns_fname)
    t_start = 0
    t_end = stim_times[-1] + 200
    spike_matrix, gids, t_bins = spikes2mat(spike_times, spiking_gids, t_start, t_end, bin_size)
    assert (spike_matrix.shape[0] == np.sum(spike_matrix.any(axis=1)))

    rate = np.sum(spike_matrix, axis=0)
    rate_th = sign_rate_std(spike_times, spiking_gids, t_start, t_end, bin_size, N=100)
    t_idx = np.where(rate > np.mean(rate) + rate_th)[0]
    rate_norm = len(np.unique(spiking_gids)) * 1e-3 * bin_size
    fig_name = os.path.join(fig_path, "rate_seed%i.png" % seed)
    plot_rate(rate / rate_norm, rate_th / rate_norm, t_start, t_end, fig_name)
    spike_matrix_dict = {seed: SpikeMatrixResult(spike_matrix[:, t_idx], gids, t_bins[t_idx])}

    return spike_matrix_dict, stim_times, patterns


if __name__ == "__main__":

    N = 5000  # total number of neurons
    n = 200  # neurons in 1 assembly
    S = 60  # patterns presented
    overlap = True
    jitter = 5  # ms
    spike_prob = 0.8
    background_rate = 0.1  # Hz
    seed = 12345
    patterns_fname = os.path.join(inp_dir, "10patterns_S%i.txt" % S)
    spikes_fname = os.path.join(inp_dir, parse_filename(N, n, S, jitter, spike_prob, background_rate, seed))

    spike_matrix_dict, stim_times, patterns = get_sign_spike_matrices(patterns_fname, spikes_fname, seed, fig_path)
    clusters_dict = cluster_spikes(spike_matrix_dict, method="hierarchical",
                                   FigureArgs=FigureArgs(stim_times, patterns, None, None, fig_path))

