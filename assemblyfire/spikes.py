# -*- coding: utf-8 -*-
"""
Preprocesses spike to be used for cell assembly analysis
a la (Sasaki et al. 2006 and) Carillo-Reid et al. 2015:
Bin spikes and threshold activity by population firing rate
last modified: Thomas Delemontex, András Ecker 11.2020
"""

import os
import yaml
from tqdm import tqdm
from copy import deepcopy
from collections import namedtuple
from cached_property import cached_property
import numpy as np
import multiprocessing as mp
from bluepy.v2 import Simulation

from assemblyfire.utils import get_seeds, get_patterns, get_E_gids, get_spikes


SpikeMatrixResult = namedtuple("SpikeMatrixResult", ["spike_matrix", "row_map", "t_bins", "t_idx"])
ThresholdedRate = namedtuple("ThresholdedRate", ["rate", "rate_th"])


def spikes2mat(spike_times, spiking_gids, t_start, t_end, bin_size):
    """Bins time and builds spike matrix"""

    gids = np.unique(spiking_gids)
    gid_bins = np.hstack([sorted(gids), np.max(gids) + 1])
    t_bins = np.arange(t_start, t_end + bin_size, bin_size)
    spike_matrix = np.histogram2d(spiking_gids, spike_times, bins=(gid_bins, t_bins))[0]
    return spike_matrix, gid_bins[:-1], t_bins


def calc_rate(spike_matrix):
    """Calculates (non-normalized) firing rate from spike matrix"""
    return np.sum(spike_matrix, axis=0)


def get_rate_norm(N, bin_size):
    """Rate normalization factor (for the rate to be plotted with proper dimensions)"""
    return N * 1e-3 * bin_size


def shuffle_tbins(spike_times, spiking_gids, t_start, t_end, bin_size):
    """Creates surrogate dataset a la Sasaki et al. 2006 by randomly offseting every spike
    (thus after discretization see `spike2mat()` they will be in an other time bin)"""

    spike_times += bin_size * np.random.choice([-1, 1], len(spike_times))
    spike_matrix, _, _ = spikes2mat(spike_times, spiking_gids,
                                    t_start, t_end, bin_size)
    return spike_matrix


def _shuffle_tbins_subprocess(inputs):
    """Subprocess used by multiprocessing pool for setting significance threshold"""
    surr_spike_matrix = shuffle_tbins(*inputs)
    return np.std(calc_rate(surr_spike_matrix))


def sign_rate_std(spike_times, spiking_gids, t_start, t_end, bin_size, N=100):
    """Generates surrogate datasets, checks the std of rates
    and sets significance threshold to its 95% percentile"""

    n = N if mp.cpu_count()-1 > N else mp.cpu_count()-1
    pool = mp.Pool(processes=n)
    tbin_stds = pool.map(_shuffle_tbins_subprocess, zip([deepcopy(spike_times) for _ in range(N)],
                                                        [spiking_gids for _ in range(N)],
                                                        [t_start for _ in range(N)],
                                                        [t_end for _ in range(N)],
                                                        [bin_size for _ in range(N)]))
    pool.terminate()
    return np.percentile(tbin_stds, 95)


class SpikeMatrixGroup(object):
    """Class to store metadata about simulations, binned raster and significant time bins"""

    def __init__(self, config):
        """YAML config file based constructor"""
        with open(config, "r") as f:
            self._config = yaml.load(f, Loader=yaml.SafeLoader)

    @property
    def config(self):
        return self._config

    @property
    def root_path(self):
        return self.config["root_path"]

    @property
    def root_fig_path(self):
        return self.config["root_fig_path"]

    @property
    def t_start(self):
        return self.config["preprocessing_protocol"]["t_start"]

    @property
    def t_end(self):
        return self.config["preprocessing_protocol"]["t_end"]

    @property
    def bin_size(self):
        return self.config["preprocessing_protocol"]["bin_size"]

    @property
    def clustering_method(self):
        assert self.config["clustering_method"] in ["hierarchical", "density_based"]
        return self.config["clustering_method"]

    @cached_property
    def seeds(self):
        return get_seeds(self.root_path)

    @cached_property
    def patterns(self):
        return get_patterns(self.root_path)

    def get_blueconfig_path(self, seed):
        return os.path.join(self.root_path, "stimulusstim_a0", "seed%i" % seed, "BlueConfig")

    def get_spike_matrices(self):
        """Bin spikes and threshold activity by population firing rate"""

        spike_matrix_dict = {}; rate_dict = {}
        for seed in tqdm(self.seeds):
            sim = Simulation(self.get_blueconfig_path(seed))
            gids = get_E_gids(sim.circuit, sim.target)
            spike_times, spiking_gids = get_spikes(sim, gids, self.t_start, self.t_end)
            spike_matrix, row_map, t_bins = spikes2mat(spike_times, spiking_gids,
                                                       self.t_start, self.t_end, self.bin_size)
            assert (spike_matrix.shape[0] == np.sum(spike_matrix.any(axis=1)))

            # threshold rate: this could be a separate function...
            # but then one would need to store `spike_times` and `spiking_gids`
            rate = calc_rate(spike_matrix)
            # get sign threshold (compare to Monte-Carlo shuffles)
            rate_th = sign_rate_std(spike_times, spiking_gids,
                                    self.t_start, self.t_end, self.bin_size)
            rate_norm = get_rate_norm(len(np.unique(spiking_gids)), self.bin_size)
            # get ids of significant (above threshold) time bins
            t_idx = np.where(rate > np.mean(rate) + rate_th)[0]
            spike_matrix_dict[seed] = SpikeMatrixResult(spike_matrix, row_map, t_bins, t_idx)
            rate_dict[seed] = ThresholdedRate(rate/rate_norm, rate_th/rate_norm)

        return spike_matrix_dict, rate_dict
