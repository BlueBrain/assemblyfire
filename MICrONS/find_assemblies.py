"""
Minor modifications to `../assemblyfire/find_assemblies.py` to be able to run on spike traces
extracted from the co-registered MICrONS dataset (see `query_functional_data.py` for fetching that data)
last modified: András Ecker 09.2023
"""

import os
from collections import namedtuple
import numpy as np

from assemblyfire.utils import ensure_dir
from assemblyfire.spikes import get_sign_rate_th, spikes_to_h5
from assemblyfire.clustering import cluster_spikes, detect_assemblies
from assemblyfire.plots import plot_rate

SpikeMatrixResult = namedtuple("SpikeMatrixResult", ["spike_matrix", "gids", "t_bins"])


def bin_spikes(spikes, idx, t, bin_size, min_th=0.01):
    """Convert deconvolved Ca++ traces into assemblyfire's `spike_matrix` format"""
    assert bin_size > t[1] - t[0]
    sort_idx = np.argsort(idx)
    spikes = spikes[sort_idx, :]
    t_bins = np.arange(t[0], t[-1] + bin_size, bin_size)
    spike_matrix = np.zeros((len(idx), len(t_bins)), dtype=np.float32)
    for i, (t_start, t_end) in enumerate(zip(t_bins[:-1], t_bins[1:])):
        t_idx = np.where((t_start <= t) & (t < t_end))[0]
        spike_matrix[:, i] = np.mean(spikes[:, t_idx], axis=1)
    spike_matrix[spike_matrix < min_th] = 0.  # this will make it sparse and our life easier later
    return spike_matrix, idx[sort_idx], t_bins


if __name__ == "__main__":
    session_id, scan_id = 9, 6
    bin_size = 0.5  # s (not ms!)
    seed = 10 * session_id + scan_id  # fake seed for saving
    save_tag = "session%i_scan%i" % (session_id, scan_id)
    fig_path = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/figures/v7_assemblies/MICrONS_%s" % save_tag
    ensure_dir(fig_path)

    # bin "spikes" and get sign. time bins
    data = np.load("MICrONS_%s.npz" % save_tag)
    spike_matrix, gids, t_bins = bin_spikes(data["spikes"], data["idx"], data["t"], bin_size)
    rate = np.sum(spike_matrix, axis=0)
    rate_th = get_sign_rate_th(spike_matrix, "keep_sc_rate")
    t_idx = np.where(rate > np.mean(rate) + rate_th)[0]
    plot_rate(rate, rate_th, t_bins[0], t_bins[-1], os.path.join(fig_path, "rate_seed%i.png" % seed))
    # save spikes to assemblyfire's format
    spike_matrix_dict = {seed: SpikeMatrixResult(spike_matrix[:, t_idx], gids, t_bins[t_idx])}
    metadata = {"root_path": "MICrONS_session%i_scan%i" % (session_id, scan_id), "t": np.array([t_bins[0], t_bins[-1]]),
                "stim_times": data["stim_times"], "patterns": data["pattern_names"].tolist()}
    h5f_name = "assemblies_%s.h5" % save_tag
    spikes_to_h5(h5f_name, spike_matrix_dict, metadata, "spikes")

    # no (drastic) changes from here... everything as in `../assemblyfire/find_assemblies.py`
    clusters_dict = cluster_spikes(spike_matrix_dict, {}, metadata, fig_path)
    detect_assemblies(spike_matrix_dict, clusters_dict, 95, h5f_name, "assemblies", None, fig_path)

