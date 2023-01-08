"""
Preprocesses spike to be used for cell assembly analysis
a la (Sasaki et al. 2006 and) Carillo-Reid et al. 2015:
Bin spikes and threshold activity by population firing rate
+ SpikeMatrixGroup has some extra functionality to calculate
spike time reliability across seeds
authors: Thomas Delemontex and András Ecker; last update 01.2023
"""

import os
import gc
from tqdm import tqdm
from tqdm.contrib import tzip
from collections import namedtuple, OrderedDict
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import squareform
import multiprocessing as mp
from joblib import Parallel, delayed

from assemblyfire.config import Config

SpikeMatrixResult = namedtuple("SpikeMatrixResult", ["spike_matrix", "gids", "t_bins"])


def load_spikes(blueconfig_path, target, t_start, t_end):
    """Loads in spikes from simulations using bluepy"""
    from assemblyfire.utils import get_bluepy_simulation, get_gids, get_spikes
    sim = get_bluepy_simulation(blueconfig_path)
    gids = get_gids(sim.circuit, target)
    spike_times, spiking_gids = get_spikes(sim, gids, t_start, t_end)
    return spike_times, spiking_gids


def spikes2mat(spike_times, spiking_gids, t_start, t_end, bin_size):
    """Bins time and builds spike matrix"""
    gids = np.unique(spiking_gids)
    gid_bins = np.hstack([sorted(gids), np.max(gids) + 1])
    t_bins = np.arange(t_start, t_end + bin_size, bin_size)
    spike_matrix = np.histogram2d(spiking_gids, spike_times, bins=(gid_bins, t_bins))[0]
    return spike_matrix, gid_bins[:-1], t_bins


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


def sign_rate_std(spike_times, spiking_gids, t_start, t_end, bin_size, nreps=100):
    """Generates surrogate datasets, checks the std of rates
    and sets significance threshold to its 95% percentile"""
    nprocs = nreps if os.cpu_count() - 1 > nreps else os.cpu_count() - 1
    # joblib implementation just runs out of memory...
    pool = mp.Pool(processes=nprocs)
    tbin_stds = pool.map(_shuffle_tbins_subprocess, zip([spike_times.copy() for _ in range(nreps)],
                                                        [spiking_gids for _ in range(nreps)],
                                                        [t_start for _ in range(nreps)],
                                                        [t_end for _ in range(nreps)],
                                                        [bin_size for _ in range(nreps)]))
    pool.terminate()
    return np.percentile(tbin_stds, 95)


def convolve_spike_matrix(blueconfig_path, target, t_start, t_end, bin_size=1, std=10):
    """Bins spikes and convolves it with a 1D Gaussian kernel row-by-row
    (default bin size = 1 ms comes from Nolte et al. 2019 but the kernel's std = 10 ms is set by Daniela's tests)"""
    spike_times, spiking_gids = load_spikes(blueconfig_path, target, t_start, t_end)
    spike_matrix, gids, _ = spikes2mat(spike_times, spiking_gids, t_start, t_end, bin_size)
    assert (spike_matrix.shape[0] == np.sum(spike_matrix.any(axis=1)))
    nprocs = spike_matrix.shape[0] if os.cpu_count() - 1 > spike_matrix.shape[0] else os.cpu_count() - 1
    with Parallel(n_jobs=nprocs, prefer="threads") as p:
        convolved_spike_trains = p(delayed(gaussian_filter1d)(spike_matrix[i, :], std/bin_size, output=np.float32)
                                   for i in range(spike_matrix.shape[0]))
    del spike_matrix
    gc.collect()
    return np.vstack(convolved_spike_trains), gids


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


def single_cell_features_to_h5(h5f_name, gids, r_spikes, prefix):
    """Saves single cell features to HDF5 file"""
    with h5py.File(h5f_name, "a") as h5f:
        grp = h5f.require_group(prefix)
        grp.create_dataset("gids", data=gids)
        grp.create_dataset("r_spikes", data=r_spikes)


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

    def get_sign_spike_matrices(self, save=True):
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
        stim_times, patterns = get_stimulus_stream(self.input_patterns_fname, self.t_start, self.t_end)
        project_metadata = {"root_path": self.root_path, "seeds": seeds, "t": ts,
                            "stim_times": stim_times, "patterns": patterns.tolist()}
        if save:
            spikes_to_h5(self.h5f_name, spike_matrix_dict, project_metadata, prefix=self.h5_prefix_spikes)
        return spike_matrix_dict, project_metadata

    def get_spike_time_reliability(self):
        """Convolution based spike time reliability (`r_spike`) measure from Schreiber et al. 2003"""
        from assemblyfire.utils import get_sim_path
        from assemblyfire.clustering import cosine_similarity

        # one can't simply np.dstack() them because it's not guaranteed that all gids spike in all trials
        gid_dict, convolved_spike_matrix_dict = {}, {}
        sim_paths = get_sim_path(self.root_path)
        for seed, blueconfig_path in tqdm(sim_paths.items(), desc="Loading in simulation results"):
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
        r_spikes = np.zeros_like(all_gids, dtype=np.float32)
        for i, gid in enumerate(tqdm(all_gids, desc="Concatenating results for all gids", miniters=len(all_gids) / 100)):
            # array of single neuron across trials with <=len(seed) rows
            gid_trials_convolved = np.stack([convolved_spike_matrix_dict[seed][(gids == gid), :][0]
                                             for seed, gids in gid_dict.items()
                                             if len(convolved_spike_matrix_dict[seed][(gids == gid), :])])
            if gid_trials_convolved.shape[0] > 1:
                gid_trials_convolved -= np.mean(gid_trials_convolved, axis=1).reshape(-1, 1)  # mean center trials
                sim_matrix = cosine_similarity(gid_trials_convolved)
                # squareform implements its inverse if the input is a square matrix (but the diagonal has to be 0.)
                np.fill_diagonal(sim_matrix, 0)  # stupid numpy...
                r_spikes[i] = np.mean(squareform(sim_matrix))
        return all_gids, r_spikes
