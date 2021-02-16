# -*- coding: utf-8 -*-
"""
Generate artificial spiketrains in order to test the cell assembly analysis pipeline
last modified: András Ecker 08.2020
"""

import os
import numpy as np

out_dir = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/simulations/test_spiketrains"


def def_assemblies(assembly_names, N, n, overlap):
    """Defines cell assemblies (n neurons in each) within a population of N neurons"""

    gids = np.arange(N)
    # define assemblies
    if overlap:
        assemblies = {name: np.sort(np.random.choice(gids, n, replace=False)) for name in assembly_names}
    else:
        # make sure that 1 neuron doesn't belong to several assemblies
        assemblies = {}
        gid_idx = np.arange(len(gids))
        for name in assembly_names:
            tmp = np.arange(len(gid_idx))
            idx = np.random.choice(tmp, n, replace=False)
            assemblies[name] = np.sort(gid_idx[idx])
            gid_idx = np.delete(gid_idx, idx)

    return gids, assemblies


def get_stim_order(assembly_names, S):
    """Reads/generates the order of the patterns to present"""

    assert(len(assembly_names) == 10)
    f_name = os.path.join(out_dir, "10patterns_S%i.txt" % S)
    if os.path.isfile(f_name):
        from assemblyfire.utils import get_patterns, get_stim_times
        stim_order = get_patterns(f_name)
        stim_times = get_stim_times(f_name)
    else:
        stim_order = np.random.choice(assembly_names, S)
        stim_times = 200 + np.arange(S) * 200  # 1 pattern presented in every 200 ms
        with open(f_name, "w") as f:
            for t, pattern_name in zip(stim_times, stim_order):
                f.write("%i %s\n" % (t, pattern_name))
    return stim_order, stim_times


def parse_filename(N, n, S, jitter, spike_prob, background_rate, seed):
    return "N%i_n%i_S%i_j%i_sp%.1f_Hz%.1f_seed%i.out.dat" % (N, n, S, jitter, spike_prob, background_rate, seed)


def _exp_rand(lambda_, n, seed):
    """MATLAB style random exponential numbers"""
    np.random.seed(seed)
    return -1.0 / lambda_ * np.log(np.random.rand(n))


def poisson_proc(lambda_, n, t_max, seed):
    """Poisson process (exp. distributed ISIs)"""

    rnd_isis = _exp_rand(lambda_, int(n), seed)
    poisson_proc = np.cumsum(rnd_isis)
    assert poisson_proc[-1] > t_max, "Spike train is too short, consider increasing `n`!"
    return poisson_proc[np.where(poisson_proc <= t_max)]


def gen_spiktrain(assemblies, stim_order, stim_times, jitter, spike_prob, background_rate, f_name):
    """Generate test spiketrain: background activity + synch. activation of assembly neurons
    (with given spike_prob and some jitter)"""

    # generate spike times for assemblies
    spiking_gids = assemblies[stim_order[0]]
    spike_times = stim_times[0]+jitter + 2*jitter*np.random.sample(len(spiking_gids))-jitter
    for stim_time, name in zip(stim_times[1:], stim_order[1:]):
        assembly = assemblies[name]
        spiking_gids = np.concatenate((spiking_gids, assembly))
        tmp = stim_time+jitter + 2*jitter*np.random.sample(len(assembly))-jitter
        spike_times = np.concatenate((spike_times, tmp))

    # mask some spikes based on spike prob.
    if spike_prob < 1.:
        np.random.seed(seed)
        idx = np.where(np.random.random_sample(len(spike_times)) < spike_prob)
        spiking_gids = spiking_gids[idx]
        spike_times = spike_times[idx]

    # generate background spiking activity (might take some time)
    t_end = 0.2 + stim_times[-1]/1000.
    if background_rate > 0.:
        for i in gids:
            tmp = poisson_proc(background_rate, 10, t_end, seed+i) * 1000  # ms conversion
            spike_times = np.concatenate((spike_times, tmp))
            spiking_gids = np.concatenate((spiking_gids, i*np.ones_like(tmp)))

        # reorder spike times to have the same format as neurodamus' out.dat
        sort_idx = np.argsort(spike_times)
        spike_times = spike_times[sort_idx]
        spiking_gids = spiking_gids[sort_idx]
        assert(spiking_gids.shape == spike_times.shape)

    # save to file
    with open(f_name, "w") as f:
        f.write("\scatter\n")
        for t, gid in zip(spike_times, spiking_gids):
            f.write("%.2f %i\n" % (t, gid))


if __name__ == "__main__":

    N = 5000  # total number of neurons
    n = 200  # neurons in 1 assembly
    S = 60  # patterns presented
    overlap = True
    jitter = 5  # ms
    spike_prob = 0.8
    background_rate = 0.1  # Hz
    seed = 12345
    assembly_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]  # 10 assemblies
    f_name = os.path.join(out_dir, parse_filename(N, n, S, jitter, spike_prob, background_rate, seed))

    gids, assemblies = def_assemblies(assembly_names, N, n, overlap)
    stim_order, stim_times = get_stim_order(assembly_names, S)
    gen_spiktrain(assemblies, stim_order, stim_times, jitter, spike_prob, background_rate, f_name)
