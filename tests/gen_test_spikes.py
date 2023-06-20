"""
Generate artificial spiketrains in order to test the cell assembly analysis pipeline
last modified: András Ecker 06.2022
"""

import os
import h5py
import numpy as np
import pandas as pd

from assemblyfire.config import Config


def def_assemblies(pattern_names, N, n, overlap):
    """Defines cell assemblies (one for each pattern, and `n` neurons in each) within a population of `N` neurons"""
    gids = np.arange(N)
    if overlap:
        assemblies = {name: np.sort(np.random.choice(gids, n, replace=False)) for name in pattern_names}
    else:  # make sure that 1 neuron doesn't belong to several assemblies
        assemblies = {}
        gid_idx = np.arange(len(gids))
        for name in pattern_names:
            tmp = np.arange(len(gid_idx))
            idx = np.random.choice(tmp, n, replace=False)
            assemblies[name] = np.sort(gid_idx[idx])
            gid_idx = np.delete(gid_idx, idx)
    return gids, assemblies


def get_stim_order(f_name, pattern_names, t_start, t_end, isi):
    """Reads/generates the order of the patterns to present"""
    if os.path.isfile(f_name):
        from assemblyfire.utils import get_stimulus_stream
        stim_times, stim_order = get_stimulus_stream(f_name)
    else:
        n = int((t_end - t_start) / isi)
        stim_times = isi + np.arange(n) * isi
        stim_order = np.random.choice(pattern_names, n)

        with open(f_name, "w") as f:
            for t, pattern_name in zip(stim_times, stim_order):
                f.write("%i %s\n" % (t, pattern_name))
    return stim_times, stim_order


def _generate_exp_rand_numbers(lambda_, n_rnds, seed):
    """MATLAB's random exponential number"""
    np.random.seed(seed)
    return -1.0 / lambda_ * np.log(np.random.rand(n_rnds))


def generate_hom_poisson_spikes(rate, t_start, t_end, seed):
    """Generates Poisson process: (interval times are exponentially distributed:
    X_i = -ln(U_i)/lambda_, where lambda_ is the rate and U_i ~ Uniform(0,1))
    (`rate` in Hz, `t_start` and `t_end` in ms)"""
    expected_n = (t_end - t_start) / 1000. * rate
    n_rnds = np.ceil(expected_n + 3 * np.sqrt(expected_n))  # NeuroTools' way of determining the number of random ISIs
    rnd_isis = _generate_exp_rand_numbers(rate, int(n_rnds), seed) * 1000.  # ISIs in ms
    poisson_proc = np.cumsum(rnd_isis) + t_start
    if poisson_proc[-1] > t_end:
        return poisson_proc[poisson_proc <= t_end]
    else:
        i, extra_spikes = 1, []
        t_last = poisson_proc[-1] + _generate_exp_rand_numbers(rate, 1, seed+i)[0]
        while t_last < t_end:
            extra_spikes.append(t_last)
            i += 1
            t_last += _generate_exp_rand_numbers(rate, 1, seed+i)[0]
        return np.concatenate((poisson_proc, extra_spikes))


def gen_spiktrain(assemblies, stim_order, stim_times, jitter, spike_prob, background_rate):
    """Generate test spiketrain: background activity + synch. activation of assembly neurons
    (with given spike_prob and some jitter)"""
    # generate spike times for assemblies
    spike_times, spiking_gids = [], []
    for stim_time, pattern_name in zip(stim_times, stim_order):
        assembly_gids = assemblies[pattern_name]
        spiking_gids.append(assembly_gids)
        spike_times.append(stim_time + jitter + 2 * jitter * np.random.sample(len(assembly_gids)) - jitter)
    spiking_gids = np.concatenate(spiking_gids)
    spike_times = np.concatenate(spike_times)

    if spike_prob < 1.:  # mask some spikes based on spike prob.
        np.random.seed(seed)
        idx = np.where(np.random.random_sample(len(spike_times)) < spike_prob)
        spiking_gids = spiking_gids[idx]
        spike_times = spike_times[idx]

    if background_rate > 0.:  # generate background spiking activity
        t_start, t_end = 0, stim_times[-1]
        bg_spike_times, bg_spiking_gids = [], []
        for gid in gids:
            tmp = generate_hom_poisson_spikes(background_rate, t_start, t_end, seed+gid)
            bg_spike_times.append(tmp)
            bg_spiking_gids.append(gid * np.ones_like(tmp))
        spike_times = np.concatenate([spike_times, np.concatenate(bg_spike_times)])
        spiking_gids = np.concatenate([spiking_gids, np.concatenate(bg_spiking_gids)])
        sort_idx = np.argsort(spike_times)
        spike_times = spike_times[sort_idx]
        spiking_gids = spiking_gids[sort_idx]

    return spike_times, spiking_gids


def save_spikes(h5f_name, prefix, spike_times, spiking_gids):
    """Save spikes to SONATA format"""
    assert (spiking_gids.shape == spike_times.shape)
    with h5py.File(h5f_name, "w") as h5f:
        grp = h5f.require_group("spikes/%s" % prefix)
        # grp.attrs["sorting"] = 2  # this should be an enum, but it works w/o it as well
        grp.create_dataset("timestamps", data=spike_times)
        grp["timestamps"].attrs["units"] = "ms"
        grp.create_dataset("node_ids", data=spiking_gids, dtype=int)


if __name__ == "__main__":
    N = 5000  # total number of neurons
    n = 200  # neurons in 1 assembly
    isi = 200  # ms (between pattern presentations)
    overlap = True
    jitter = 5  # ms
    spike_prob = 0.8
    background_rate = 0.1  # Hz
    seed = 12345

    config = Config("test.yaml")
    pattern_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    stim_times, stim_order = get_stim_order(config.input_sequence_fname,
                                            pattern_names, config.t_start, config.t_end, isi)
    gids, assemblies = def_assemblies(pattern_names, N, n, overlap)
    spike_times, spiking_gids = gen_spiktrain(assemblies, stim_order, stim_times, jitter, spike_prob, background_rate)
    f_name = os.path.join(config.root_path, "spikes.h5")
    save_spikes(f_name, config.node_pop, spike_times, spiking_gids)
    spike_paths = pd.Series(data=f_name, index=[seed])
    spike_paths.index.name = "seed"
    spike_paths.to_pickle(os.path.join(config.root_path, "analyses", "simulations.pkl"))
