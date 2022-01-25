"""
Preprocesses spike to be used for cell assembly analysis
a la (Sasaki et al. 2006 and) Carillo-Reid et al. 2015:
Bin spikes and threshold activity by population firing rate
+ SpikeMatrixGroup has some extra functionality to calculate
spike time reliability across seeds
authors: Thomas Delemontex and András Ecker; last update 01.2022
"""

import os
import gc
from tqdm import tqdm
from tqdm.contrib import tzip
from copy import deepcopy
from collections import namedtuple, OrderedDict
import h5py
import numpy as np
import multiprocessing as mp

from assemblyfire.config import Config

SpikeMatrixResult = namedtuple("SpikeMatrixResult", ["spike_matrix", "gids", "t_bins"])


def load_spikes(blueconfig_path, target, t_start, t_end):
    """Loads in spikes from simulations using bluepy"""
    from assemblyfire.utils import get_bluepy_simulation, get_E_gids, get_spikes
    sim = get_bluepy_simulation(blueconfig_path)
    gids = get_E_gids(sim.circuit, target)
    spike_times, spiking_gids = get_spikes(sim, gids, t_start, t_end)
    return spike_times, spiking_gids


def spikes2mat(spike_times, spiking_gids, t_start, t_end, bin_size):
    """Bins time and builds spike matrix"""
    gids = np.unique(spiking_gids)
    gid_bins = np.hstack([sorted(gids), np.max(gids) + 1])
    t_bins = np.arange(t_start, t_end + bin_size, bin_size)
    spike_matrix = np.histogram2d(spiking_gids, spike_times, bins=(gid_bins, t_bins))[0]
    return spike_matrix, gid_bins[:-1], t_bins


def get_ts_in_bin(spike_times, spiking_gids, bin_size):
    """Gets time in bin for all gids"""
    idx_sort = np.argsort(spiking_gids)
    spiking_gids = spiking_gids[idx_sort]
    unique_gids, idx_start, counts = np.unique(spiking_gids, return_index=True, return_counts=True)
    ts = {}
    spike_times = spike_times[idx_sort]
    for gid, id_start, count in zip(unique_gids, idx_start, counts):
        ts[gid] = np.mod(spike_times[id_start:id_start+count], bin_size)
    return ts


def calc_rate(spike_matrix):
    """Calculates (non-normalized) firing rate from spike matrix"""
    return np.sum(spike_matrix, axis=0)


def get_rate_norm(N, bin_size):
    """Rate normalization factor (for the rate to be plotted with proper dimensions)"""
    return N * 1e-3 * bin_size


def shuffle_tbins(spike_times, spiking_gids, t_start, t_end, bin_size):
    """Creates surrogate dataset a la Sasaki et al. 2006 by randomly offsetting every spike
    (thus after discretization see `spike2mat()` they will be in another time bin)"""
    spike_times += bin_size * np.random.choice([-1, 1], len(spike_times))
    spike_matrix, _, _ = spikes2mat(spike_times, spiking_gids, t_start, t_end, bin_size)
    return spike_matrix


def _shuffle_tbins_subprocess(inputs):
    """Subprocess used by multiprocessing pool for setting significance threshold"""
    surr_spike_matrix = shuffle_tbins(*inputs)
    return np.std(np.sum(surr_spike_matrix, axis=0))


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


def convolve_spike_matrix(blueconfig_path, target, t_start, t_end, bin_size=1, std=0.5):
    """Bins spikes and convolves it with a 1D Gaussian kernel row-by-row
    (default bin size = 1 ms and kernel's std = 0.5 ms comes from Nolte et al. 2019)"""
    spike_times, spiking_gids = load_spikes(blueconfig_path, target, t_start, t_end)
    spike_matrix, gids, _ = spikes2mat(spike_times, spiking_gids, t_start, t_end, bin_size)
    assert (spike_matrix.shape[0] == np.sum(spike_matrix.any(axis=1)))
    x = np.linspace(0, 10 * std, 11)  # 11 is hard coded... feel free to find a better number
    kernel = np.exp(-np.power(x - 5 * std, 2) / (2 * std ** 2))
    convolved_spike_matrix = np.zeros_like(spike_matrix, dtype=np.float32)
    for i in range(spike_matrix.shape[0]):
        convolved_spike_matrix[i, :] = np.convolve(spike_matrix[i, :], kernel, mode="same")
    return convolved_spike_matrix, gids


def spikes_to_h5(h5f_name, spike_matrix_dict, metadata, prefix):
    """Saves spike matrices to HDF5 file"""
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
    """Saves single cell features to HDF5 file"""
    with h5py.File(h5f_name, "a") as h5f:
        grp = h5f.require_group(prefix)
        grp.create_dataset("gids", data=gids)
        grp.create_dataset("r_spikes", data=r_spikes)
        grp.create_dataset("mean_ts_in_bin", data=mean_ts_in_bin)
        grp.create_dataset("std_ts_in_bin", data=std_ts_in_bin)


class SpikeMatrixGroup(Config):
    """Class that bins rasters, finds significant time bins and extract single cell features"""

    def get_sign_spike_matrix(self, blueconfig_path, t_start, t_end):
        """Bins spikes and thresholds population firing rate to get significant time bins"""
        spike_times, spiking_gids = load_spikes(blueconfig_path, self.target, t_start, t_end)
        spike_matrix, gids, t_bins = spikes2mat(spike_times, spiking_gids, t_start, t_end, self.bin_size)
        assert (spike_matrix.shape[0] == np.sum(spike_matrix.any(axis=1)))
        rate = np.sum(spike_matrix, axis=0)
        if self.threshold_rate:
            rate_th = sign_rate_std(spike_times, spiking_gids, t_start, t_end, self.bin_size)
            t_idx = np.where(rate > np.mean(rate) + rate_th)[0]  # get ids of significant time bins
            spike_matrix_results = SpikeMatrixResult(spike_matrix[:, t_idx], gids, t_bins[t_idx])
        else:
            rate_th, spike_matrix_results = np.nan, SpikeMatrixResult(spike_matrix, gids, t_bins)
        rate_norm = len(np.unique(spiking_gids)) * 1e-3 * self.bin_size
        return spike_matrix_results, rate / rate_norm, rate_th / rate_norm

    def get_sign_spike_matrices(self):
        """Looped version of `get_sign_spike_matrix()` above for all conditions in the campaign"""
        from assemblyfire.utils import get_sim_path, get_stimulus_stream
        from assemblyfire.plots import plot_rate

        spike_matrix_dict = OrderedDict()
        sim_paths = get_sim_path(self.root_path)
        if self.t_chunks is None:
            ts = np.array([self.t_start, self.t_end])
            seeds = np.sort(sim_paths.index.to_numpy())
            for seed in tqdm(seeds, desc="Loading in simulation results"):
                if seed in self.ignore_seeds:
                    pass
                spike_matrix, rate, rate_th = self.get_sign_spike_matrix(sim_paths.loc[seed], self.t_start, self.t_end)
                spike_matrix_dict[seed] = spike_matrix
                del spike_matrix
                gc.collect()
                fig_name = os.path.join(self.fig_path, "rate_seed%i.png" % seed)
                plot_rate(rate, rate_th, self.t_start, self.t_end, fig_name)
        else:
            assert (len(sim_paths) == 1), "Chunking sim only works for a single seed is atm."
            ts = np.linspace(self.t_start, self.t_end, self.t_chunks+1)
            seeds = np.arange(self.t_chunks)  # chunks will still be referenced as "seed" for convenience
            for seed, t_start, t_end in tzip(seeds, ts[:-1], ts[1:], desc="Loading in simulation results"):
                spike_matrix, rate, rate_th = self.get_sign_spike_matrix(sim_paths.iloc[0], t_start, t_end)
                spike_matrix_dict[seed] = spike_matrix
                del spike_matrix
                gc.collect()
                fig_name = os.path.join(self.fig_path, "rate_seed%i.png" % seed)
                plot_rate(rate, rate_th, t_start, t_end, fig_name)
        stim_times, patterns = get_stimulus_stream(self.patterns_fname, self.t_start, self.t_end)
        project_metadata = {"root_path": self.root_path, "seeds": seeds, "t": ts,
                            "stim_times": stim_times, "patterns": patterns.tolist()}
        spikes_to_h5(self.h5f_name, spike_matrix_dict, project_metadata, prefix=self.h5_prefix_spikes)
        return spike_matrix_dict, project_metadata

    def get_mean_std_ts_in_bin(self):
        """Gets mean and std of spike times within the bins for all gids across seeds"""
        from assemblyfire.utils import get_sim_path

        # load in spikes and get times in bin per seed
        indiv_gids, ts_in_bin = [], {}
        sim_paths = get_sim_path(self.root_path)
        for seed, blueconfig_path in tqdm(sim_paths.iteritems(), desc="Loading in simulation results"):
            if seed in self.ignore_seeds:
                pass
            spike_times, spiking_gids = load_spikes(blueconfig_path, self.target, self.t_start, self.t_end)
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

    def get_spike_time_reliability(self):
        """Convolution based spike time reliability (`r_spike`) measure from Schreiber et al. 2003"""
        from scipy.spatial.distance import pdist
        from assemblyfire.utils import get_sim_path

        # one can't simply np.dstack() them because it's not guaranteed that all gids spike in all trials
        gid_dict, convolved_spike_matrix_dict = {}, {}
        sim_paths = get_sim_path(self.root_path)
        for seed, blueconfig_path in tqdm(sim_paths.iteritems(), desc="Loading in simulation results"):
            if seed in self.ignore_seeds:
                pass
            convolved_spike_matrix, gids = convolve_spike_matrix(blueconfig_path, self.target, self.t_start, self.t_end)
            convolved_spike_matrix_dict[seed], gid_dict[seed] = convolved_spike_matrix, gids
            del convolved_spike_matrix
            gc.collect()
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
