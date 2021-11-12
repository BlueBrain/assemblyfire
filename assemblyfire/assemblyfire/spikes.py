# -*- coding: utf-8 -*-
"""
Preprocesses spike to be used for cell assembly analysis
a la (Sasaki et al. 2006 and) Carillo-Reid et al. 2015:
Bin spikes and threshold activity by population firing rate
+ SpikeMatrixGroup has some extra functionality to calculate
spike time reliability across seeds
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

SpikeMatrixResult = namedtuple("SpikeMatrixResult", ["spike_matrix", "gids", "t_bins"])


def get_bluepy_simulation(blueconfig_path):
    try:
        from bluepy.v2 import Simulation
    except ImportError as e:
        msg = (
            "Assemblyfire requirements are not installed.\n"
            "Please pip install bluepy as follows:\n"
            " pip install -i https://bbpteam.epfl.ch/repository/devpi/simple bluepy[all]"
        )
        raise ImportError(str(e) + "\n\n" + msg)
    return Simulation(blueconfig_path)


def spikes2mat(spike_times, spiking_gids, t_start, t_end, bin_size):
    """Bins time and builds spike matrix"""
    gids = np.unique(spiking_gids)
    gid_bins = np.hstack([sorted(gids), np.max(gids) + 1])
    t_bins = np.arange(t_start, t_end + bin_size, bin_size)
    spike_matrix = np.histogram2d(spiking_gids, spike_times, bins=(gid_bins, t_bins))[0]
    return spike_matrix, gid_bins[:-1], t_bins


def get_ts_in_bin(spike_times, spiking_gids, bin_size):
    """Gets time in bin for all gids"""
    return {gid: np.mod(spike_times[spiking_gids == gid], bin_size)
            for gid in np.unique(spiking_gids)}


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

    n = N if mp.cpu_count() - 1 > N else mp.cpu_count() - 1
    pool = mp.Pool(processes=n)
    tbin_stds = pool.map(_shuffle_tbins_subprocess, zip([deepcopy(spike_times) for _ in range(N)],
                                                        [spiking_gids for _ in range(N)],
                                                        [t_start for _ in range(N)],
                                                        [t_end for _ in range(N)],
                                                        [bin_size for _ in range(N)]))
    pool.terminate()
    return np.percentile(tbin_stds, 95)


def spike_train_convolution(spike_matrix, std):
    """Convolve spike matrix (row-by-row) with Gaussian kernel"""
    x = np.linspace(0, 10 * std, 11)  # 11 is hard coded... feel free to find a better number
    kernel = np.exp(-np.power(x - 5 * std, 2) / (2 * std ** 2))
    return np.stack(np.convolve(spike_matrix[i, :], kernel, mode="same")
                    for i in range(spike_matrix.shape[0]))


def spikes_to_h5(h5f_name, spike_matrix_dict, metadata, prefix):
    """Saves spike matrices to HDF5 file"""
    import h5py
    with h5py.File(h5f_name, "a") as h5f:
        grp = h5f.require_group(prefix)
        for k, v in metadata.items():
            grp.attrs[k] = v
        for seed, SpikeMatrixResult in spike_matrix_dict.items():
            grp_out = grp.create_group("seed%i" % seed)
            grp_out.create_dataset("spike_matrix", data=SpikeMatrixResult.spike_matrix, compression="gzip")
            grp_out.create_dataset("gids", data=SpikeMatrixResult.gids)
            grp_out.create_dataset("t_bins", data=SpikeMatrixResult.t_bins)


def single_cell_features_to_h5(h5f_name, gids, r_spikes, mean_ts_in_bin, std_ts_in_bin, prefix):
    """Saves spike matrices to HDF5 file"""
    import h5py
    with h5py.File(h5f_name, "a") as h5f:
        grp = h5f.require_group(prefix)
        grp.create_dataset("gids", data=gids)
        grp.create_dataset("r_spikes", data=r_spikes)
        grp.create_dataset("mean_ts_in_bin", data=mean_ts_in_bin)
        grp.create_dataset("std_ts_in_bin", data=std_ts_in_bin)


class SpikeMatrixGroup(object):
    """Class to store config parameters about simulations, binned raster and significant time bins"""

    def __init__(self, config_path):
        """YAML config file based constructor"""
        self._config_path = config_path
        with open(config_path, "r") as f:
            self._config = yaml.load(f, Loader=yaml.SafeLoader)

    @property
    def config(self):
        return self._config

    @property
    def root_path(self):
        return self.config["root_path"]

    @property
    def patterns_fname(self):
        return self.config["patterns_fname"]

    @property
    def h5f_name(self):
        return self.config["h5_out"]["file_name"]

    @property
    def h5_prefix_spikes(self):
        return self.config["h5_out"]["prefixes"]["spikes"]

    @property
    def h5_prefix_assemblies(self):
        return self.config["h5_out"]["prefixes"]["assemblies"]

    @property
    def root_fig_path(self):
        return self.config["root_fig_path"]

    @cached_property
    def fig_path(self):
        return os.path.join(self.root_fig_path, self._config_path.split('/')[-1][:-5])

    @property
    def target(self):
        return self.config["preprocessing_protocol"]["target"]

    @property
    def t_start(self):
        return self.config["preprocessing_protocol"]["t_start"]

    @property
    def t_start_exception(self):
        if "t_start_exception" in self.config["preprocessing_protocol"]:
            return self.config["preprocessing_protocol"]["t_start_exception"]
        else:
            return self.t_start

    @property
    def seeds_exception(self):
        if "seeds_exception" in self.config["preprocessing_protocol"]:
            return self.config["preprocessing_protocol"]["seeds_exception"]
        else:
            return []

    @property
    def ignore_seeds(self):
        if "ignore_seeds" in self.config["preprocessing_protocol"]:
            return self.config["preprocessing_protocol"]["ignore_seeds"]
        else:
            return []

    @property
    def t_end(self):
        return self.config["preprocessing_protocol"]["t_end"]

    @property
    def bin_size(self):
        return self.config["preprocessing_protocol"]["bin_size"]

    @property
    def threshold_rate(self):
        return self.config["preprocessing_protocol"]["threshold_rate"]

    @property
    def clustering_method(self):
        assert self.config["clustering_methods"]["spikes"] in ["hierarchical", "density_based"]
        return self.config["clustering_methods"]["spikes"]

    @cached_property
    def seeds(self):
        from assemblyfire.utils import get_seeds
        return get_seeds(self.root_path)

    @cached_property
    def stim_times(self):
        from assemblyfire.utils import get_stim_times
        return get_stim_times(self.patterns_fname)

    @cached_property
    def patterns(self):
        from assemblyfire.utils import get_patterns
        return get_patterns(self.patterns_fname)

    def get_blueconfig_path(self, seed):
        return os.path.join(self.root_path, "stimulusstim_a0", "seed%i" % seed, "BlueConfig")
        # return os.path.join(self.root_path, "seed%i" % seed, "BlueConfig")  # changed to analyse Sirio's sims

    def load_spikes(self, seed, t_start):
        """Loads in spikes from simulations using bluepy"""
        from assemblyfire.utils import get_E_gids, get_spikes
        sim = get_bluepy_simulation(self.get_blueconfig_path(seed))
        gids = get_E_gids(sim.circuit, self.target)
        spike_times, spiking_gids = get_spikes(sim, gids, t_start, self.t_end)
        return spike_times, spiking_gids

    def get_sign_spike_matrices(self):
        """Bin spikes and threshold activity by population firing rate to get significant time bins"""
        from assemblyfire.plots import plot_rate

        spike_matrix_dict = {}
        for seed in tqdm(self.seeds, desc="Loading in simulation results"):
            if seed in self.ignore_seeds:
                pass
            t_start = self.t_start if seed not in self.seeds_exception else self.t_start_exception
            spike_times, spiking_gids = self.load_spikes(seed, t_start)
            spike_matrix, gids, t_bins = spikes2mat(spike_times, spiking_gids,
                                                    t_start, self.t_end, self.bin_size)
            assert (spike_matrix.shape[0] == np.sum(spike_matrix.any(axis=1)))
            rate = calc_rate(spike_matrix)

            if self.threshold_rate:
                # this could be a separate function but then one would need to store `spike_times` and `spiking_gids`
                rate_th = sign_rate_std(spike_times, spiking_gids,
                                        t_start, self.t_end, self.bin_size)
                # get ids of significant (above threshold) time bins
                t_idx = np.where(rate > np.mean(rate) + rate_th)[0]
                spike_matrix_dict[seed] = SpikeMatrixResult(spike_matrix[:, t_idx], gids, t_bins[t_idx])
            else:
                rate_th = np.nan
                spike_matrix_dict[seed] = SpikeMatrixResult(spike_matrix, gids, t_bins)

            # plotting rate
            rate_norm = get_rate_norm(len(np.unique(spiking_gids)), self.bin_size)
            fig_name = os.path.join(self.fig_path, "rate_seed%i.png" % seed)
            plot_rate(rate/rate_norm, rate_th/rate_norm, t_start, self.t_end, fig_name)

        # save spikes to h5
        metadata = {"root_path": self.root_path, "seeds": self.seeds,
                    "stim_times": self.stim_times, "patterns": self.patterns}
        spikes_to_h5(self.h5f_name, spike_matrix_dict, metadata, prefix=self.h5_prefix_spikes)

        return spike_matrix_dict

    def get_mean_std_ts_in_bin(self):
        """Gets mean and std of spike times within the bins for all gids across seeds"""

        # load in spikes and get times in bin per seed
        indiv_gids = []
        ts_in_bin = {}
        for seed in tqdm(self.seeds, desc="Loading in simulation results"):
            t_start = self.t_start if seed not in self.seeds_exception else self.t_start_exception
            spike_times, spiking_gids = self.load_spikes(seed, t_start)
            indiv_gids.extend(np.unique(spiking_gids).tolist())
            ts_in_bin[seed] = get_ts_in_bin(spike_times, spiking_gids, self.bin_size)

        # iterate over all gids, concatenate the results and return mean and std
        mean_ts = []
        std_ts = []
        all_gids = np.unique(indiv_gids)
        for gid in tqdm(all_gids, desc="Concatenating results for all gids", miniters=len(all_gids) / 100):
            ts_in_bin_gid = []
            for _, ts_in_bin_seed in ts_in_bin.items():
                if gid in ts_in_bin_seed:
                    ts_in_bin_gid.extend(ts_in_bin_seed[gid].tolist())
            mean_ts.append(np.mean(ts_in_bin_gid))
            std_ts.append(np.std(ts_in_bin_gid))
        return all_gids, np.asarray(mean_ts), np.asarray(std_ts)

    def convolve_spike_matrix(self, seed):
        """Bins spikes and convolves it with a 1D Gaussian kernel row-by-row"""
        t_start = self.t_start if seed not in self.seeds_exception else self.t_start_exception
        spike_times, spiking_gids = self.load_spikes(seed, t_start)
        # bin size = 1 ms and kernel's std = 0.5 ms from Nolte et al. 2019
        spike_matrix, gids, _ = spikes2mat(spike_times, spiking_gids,
                                           t_start, self.t_end, bin_size=1)
        assert (spike_matrix.shape[0] == np.sum(spike_matrix.any(axis=1)))
        return spike_train_convolution(spike_matrix, std=0.5), gids

    def get_spike_time_reliability(self):
        """Convolution based spike time reliability (`r_spike`) measure from Schreiber et al. 2003"""
        from scipy.spatial.distance import pdist

        # one can't simply np.dstack() them because it's not guaranteed that all gids spike in all trials
        gid_dict = {}
        convolved_spike_matrix_dict = {}
        for seed in tqdm(self.seeds, desc="Loading in simulation results"):
            convolved_spike_matrix, gids = self.convolve_spike_matrix(seed)
            gid_dict[seed] = gids
            convolved_spike_matrix_dict[seed] = convolved_spike_matrix
        # build #gids matrices from trials and calculate pairwise correlation between rows
        indiv_gids = []
        for _, gids in gid_dict.items():
            indiv_gids.extend(gids)
        all_gids = np.unique(indiv_gids)
        r_spikes = []
        for gid in tqdm(all_gids, desc="Concatenating results for all gids", miniters=len(all_gids) / 100):
            # array of single neuron across trials with <=len(seed) rows
            gid_trials_convolved = np.stack(convolved_spike_matrix_dict[seed][(gids == gid), :][0]
                                            for seed, gids in gid_dict.items()
                                            if len(convolved_spike_matrix_dict[seed][(gids == gid), :]))
            n_trials = gid_trials_convolved.shape[0]
            if n_trials > 1:
                # lambda is the metric defined in Schreiber et al. 2003 (and used e.g. in Nolte et al. 2019)
                r_spike = np.sum(pdist(gid_trials_convolved,
                                       lambda u, v: np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))
                r_spikes.append(r_spike * 2 / (n_trials * (n_trials - 1)))
            else:
                r_spikes.append(0)
        return all_gids, np.asarray(r_spikes)
