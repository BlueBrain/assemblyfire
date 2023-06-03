"""
Functionalities to investigate the relationship between synaptc clustering and dendritic nonlinearities
1st step: writes launch scripts that rerun selected cells in BGLibPy (see `assemblyfire/rerun_single_cell.py`)
          (with extra reporting and some modifications)
2nd step: load spiking data from sims. with modified conditions (and baseline), check rates, and
          if based on the new spike times the neuron would be still part of the assembly
3rd step: load traces (recorded from all sections) and check for 'dendritic nonliarities'
          (to be concrete: threshold crossings sustained for a given time before a spike)
authors: Sirio Bolaños Puchet, András Ecker; last modified: 01.2023
"""

import os
import h5py
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from pyrle import Rle
import neurom as nm
from neurom.core.morphology import iter_sections
from neurom import NeuriteType

import assemblyfire.utils as utils
from assemblyfire.config import Config
from assemblyfire.clustering import get_core_cell_idx
from assemblyfire.plots import plot_across_conditions, plot_dend_traces


DSET_MEMBER = "member"
DSET_DEG = "degree"
DSET_CLST = "strength"
DSET_PVALUE = "pvalue"
CONDITIONS = ["baseline", "passivedend", "noNMDA"]

slurm_template = """\
#!/bin/sh
#SBATCH --job-name={job_name}
#SBATCH --account=proj96
#SBATCH --partition=prod
#SBATCH --constraint=cpu
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=2
#SBATCH --time=24:00:00
#SBATCH --mem=30g
#SBATCH --no-requeue
#SBATCH --output={log_name}

source /gpfs/bbp.cscs.ch/project/proj96/home/ecker/assemblyfire/setupenv.sh
assemblyfire -v rerun {config_path} {seed} {gid}\n
"""


def write_launchscripts(config_path, n_gids=10, p_th=0.05):
    """Loads synapse nearest neighbour distance results for each seed (that has any saved),
    in each assembly finds the top `n_gids` with the highest indegree and significant nnd. 'strength',
    and writes launchscripts for them"""
    config = Config(config_path)
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)

    sbatch_dir = os.path.join(config.root_path, "sbatch")
    utils.ensure_dir(sbatch_dir)
    with h5py.File(config.h5f_name, "r") as h5f:
        h5_keys = list(h5f.keys())
    simulated_gids = {}

    for seed, assembly_grp in assembly_grp_dict.items():
        prefix = "%s_syn_nnd" % seed
        if prefix in h5_keys:  # since this runs forever one might not have the results for all seeds
            syn_nnds = utils.load_syn_nnd_from_h5(config.h5f_name, len(assembly_grp), prefix=prefix)
            assembly_idx = syn_nnds.columns.get_level_values(0).unique().to_numpy()
            f_names, simulated_gids[seed] = [], {}
            for assembly_id in assembly_idx:
                df = syn_nnds.loc[:, [(assembly_id, DSET_MEMBER), (assembly_id, DSET_DEG),
                                      (assembly_id, DSET_CLST), (assembly_id, DSET_PVALUE)]]
                df.columns = df.columns.get_level_values(1)
                # index out assembly members with significant syn nnd. 'strength'
                df = df.loc[(df[DSET_MEMBER] == 1) & (df[DSET_CLST] > 0) & (df[DSET_PVALUE] < p_th)]
                # get the gids of the first n highest indegrees
                df = df.sort_values(DSET_DEG, ascending=False)
                gids = df.iloc[:n_gids].index.to_numpy()
                simulated_gids[seed][assembly_id] = gids

                # write SLURM script for all selected gids
                for gid in gids:
                    f_name = os.path.join(sbatch_dir, "%s_%s_a%i.batch" % (seed, assembly_id, gid))
                    f_names.append(f_name)
                    with open(f_name, "w+", encoding="latin1") as f:
                        f.write(slurm_template.format(job_name="rerun_a%i" % gid,
                                                      log_name=os.path.splitext(f_name)[0] + ".log",
                                                      config_path=config_path, seed=int(seed.split("seed")[1]), gid=gid))
            # write one big script that launches them all (for one seed)
            with open(os.path.join(sbatch_dir, "launch_%s.sh" % seed), "w") as f:
                for f_name in f_names:
                    f.write("sbatch %s\n" % f_name)

    # save all simulated gids
    with open(os.path.join(sbatch_dir, "simulated_gids.pkl"), "wb") as f:
        pickle.dump(simulated_gids, f, protocol=pickle.HIGHEST_PROTOCOL)


def _get_binned_spikes(results_dir, seed, assembly_gids, t_bins):
    """Loads spikes and bins them (the same way they are binned in the network simulation)
    for (correlation based) assembly membership test later"""
    assembly_idx, all_gids, all_spikes = [], [], []
    spike_vectors = {condition: [] for condition in CONDITIONS}
    for assembly_str, gids in assembly_gids.items():
        assembly_id = int(assembly_str.split("assembly")[1])
        for gid in gids:
            spikesf_name = os.path.join(results_dir, "%s_a%i_spikes.pkl" % (seed, gid))
            if os.path.isfile(spikesf_name):
                assembly_idx.append(assembly_id)
                all_gids.append(gid)
                spikes = pd.read_pickle(spikesf_name)
                spikes["gid"] = gid
                all_spikes.append(spikes)
                for condition in CONDITIONS:
                    spike_times = spikes.loc[spikes["condition"] == condition, "spike_times"].to_numpy()
                    binned_spikes, _ = np.histogram(spike_times, t_bins)
                    spike_vectors[condition].append(binned_spikes.reshape(1, -1))
            else:
                print("No spikes from: %s_assembly%i_a%i" % (seed, assembly_id, gid))
    spike_matrices = {condition: np.vstack(spike_vectors_) for condition, spike_vectors_ in spike_vectors.items()}
    return spike_matrices, np.array(all_gids), np.array(assembly_idx), pd.concat(all_spikes, ignore_index=True)


def analyse_spikes(config, seed, assembly_gids):
    """Loads spikes across conditions and gets their correlation with assembly time bins (from the orig. sim)
    (and the single cell rate of each neuron)"""
    results_dir = os.path.join(config.root_path, "analyses", "rerun_results")  # defined before but whatever...
    t_bins = np.arange(config.t_start, config.t_end + config.bin_size, config.bin_size)
    rate_norm = (config.t_end - config.t_start) / 1e3  # ms -> s conversion
    clust_meta = utils.read_cluster_seq_data(config.h5f_name)

    # build spike matrices for checking assembly membership
    spike_matrices, gids, assembly_idx, spikes = _get_binned_spikes(results_dir, seed, assembly_gids, t_bins)
    df_columns = ["gid", "assembly", "rate", "correlation", "member", "condition"]
    dfs = []
    for condition, spike_matrix in spike_matrices.items():
        rates = np.sum(spike_matrix, axis=1) / rate_norm
        idx = rates == 0
        if idx.any():
            for gid, assembly_id in zip(gids[idx], assembly_idx[idx]):
                print("No %s spikes from: %s_assembly%i_a%i" % (condition, seed, assembly_id, gid))
        # index out only significant time bins (detected before, only loaded now)
        t_idx = np.in1d(t_bins[:-1], clust_meta["t_bins"][seed], assume_unique=True)
        # check if gid would be still member of assembly
        core_cell_idx, corrs = get_core_cell_idx(spike_matrix[:, t_idx], clust_meta["clusters"][seed],
                                                 config.core_cell_th_pct)
        memberships = core_cell_idx[np.arange(len(assembly_idx)), assembly_idx].reshape(-1, 1)
        correlations = corrs[np.arange(len(assembly_idx)), assembly_idx].reshape(-1, 1)
        # put results into some meaningful format...
        df = pd.DataFrame(data=np.hstack([gids.reshape(-1, 1), assembly_idx.reshape(-1, 1), rates.reshape(-1, 1),
                                          correlations, memberships]), columns=df_columns[:-1])
        df = df.astype({"gid": int, "assembly": int, "member": int})
        df["condition"] = condition
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    # due to the random shuffled controls it can happen that neurons that were members of the assembly before,
    # won't be any longer, and we'll exclude those from the analysis...
    drop_gids = df.loc[(df["condition"] == "baseline") & (df["member"] == 0), "gid"].to_numpy()
    if len(drop_gids):
        df.drop(df.loc[df["gid"].isin(drop_gids)].index, inplace=True)
        spikes.drop(spikes.loc[spikes["gid"].isin(drop_gids)].index, inplace=True)
    # drop gids that don't spike in modified conditions (probably due to bad e-model)
    drop_gids = np.unique(df.loc[df["rate"] == 0, "gid"].to_numpy())
    if len(drop_gids):
        df.drop(df.loc[df["gid"].isin(drop_gids)].index, inplace=True)
        spikes.drop(spikes.loc[spikes["gid"].isin(drop_gids)].index, inplace=True)
    return df, spikes.drop_duplicates().sort_values("spike_times")


def _find_consecutive_ranges(peak_idx):
    """Finds consecutive windows as the name suggests... (using Rle in Cython)"""
    rle = Rle(np.diff(peak_idx) == 1)
    idx = np.where(rle.values == 1)
    start = np.insert(np.cumsum(rle.runs), 0, 0)[idx]
    count = (rle.runs + 1)[idx]
    return start, count


def trace_windows(traces_df, spike_times, threshold, sustain_for, pre_spike):
    """Finds consecutive time windows (and corresponding sections) where voltage is above a `threshold`
    for a given time (`sustained_for`) before a spike (`pre_spike`)"""
    time, secmap = np.array(traces_df.index), traces_df.columns.to_numpy()
    traces = traces_df.to_numpy()
    peaks = np.where(traces > threshold)
    peak_sec = np.unique(peaks[1])

    windows = {}
    for sec in peak_sec:  # iterate over morph sections
        peak_idx = peaks[0][np.where(peaks[1] == sec)]  # row indices where trace above threshold
        if len(peak_idx) <= 1:
            continue
        start, count = _find_consecutive_ranges(peak_idx)  # consecutive ranges in the previous array
        t_start = time[peak_idx[start]]
        t_end = time[peak_idx[start + (count - 1)]]
        deltas = np.repeat(spike_times.reshape(-1, 1), t_start.shape[0], axis=1) - t_start
        preceding = np.sum((deltas >= 0) & (deltas <= pre_spike), axis=0) > 0  # check preceding at least one spike
        long_enough = (t_end - t_start) >= sustain_for
        selected_events = np.where(preceding & long_enough)[0]  # selected events satisfy both conditions
        for i in selected_events:
            win_idx = peak_idx[range(start[i], start[i] + count[i])]
            win_trace = np.vstack((time[win_idx], traces[win_idx, sec])).transpose()
            windows.setdefault(secmap[sec], []).append(win_trace)
    return windows


def analyse_traces(results_dir, seed, spikes, threshold, sustain_for, pre_spike):
    """Loads voltage traces (from all sections) across conditions and looks for
    sustained threshold crossing before spikes"""
    dfs, df_columns = [], ["gid", "section", "start", "duration", "condition"]
    for gid in tqdm(spikes["gid"].unique(), desc="Analysing voltage traces"):
        conditions = {condition: None for condition in CONDITIONS}
        for condition in conditions.keys():
            pklf_name = os.path.join(results_dir, "%s_a%i_%s_voltages.pkl" % (seed, gid, condition))
            traces = pd.read_pickle(pklf_name)  # this takes quite a while...
            dend_traces = traces.filter(regex="(dend|apic)\[[0-9]+\]", axis=1)
            spike_times = spikes.loc[(spikes["gid"] == gid) & (spikes["condition"] == condition), "spike_times"].to_numpy()
            conditions[condition] = trace_windows(dend_traces, spike_times, threshold, sustain_for, pre_spike)

        # concatenate the results to some understandable format...
        summary = {condition: [{"section": k, "time": w[0, 0], "duration": w[-1, 0] - w[0, 0]}
                               for k, v in results.items() for w in v]
                   for condition, results in conditions.items() if results is not None}
        df = pd.DataFrame(columns=df_columns)
        i = 0
        for condition, results in summary.items():
            for w in results:
                df.loc[i, :] = [gid, w["section"], w["time"], w["duration"], condition]
                i += 1
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def get_dendritic_path_lengths(morphf_name):
    """Gets path lengths of sections using NeuroM (used for ordering plots)"""
    m = nm.load_morphology(morphf_name)
    basal_flt = lambda n: n.type == NeuriteType.basal_dendrite
    basal_df = pd.DataFrame(columns=["section", "path_length"])
    for i, sec in enumerate(iter_sections(m, neurite_filter=basal_flt, neurite_order=nm.NeuriteIter.NRN)):
        basal_df.loc[i, :] = ["dend[%i]" % i, nm.features.section.section_path_length(sec)]
    apical_flt = lambda n: n.type == NeuriteType.apical_dendrite
    apical_df = pd.DataFrame(columns=["section", "path_length"])
    for i, sec in enumerate(iter_sections(m, neurite_filter=apical_flt, neurite_order=nm.NeuriteIter.NRN)):
        apical_df.loc[i, :] = ["apic[%i]" % i, nm.features.section.section_path_length(sec)]
    return basal_df.sort_values("path_length"), apical_df.sort_values("path_length")


def plot_events(results_dir, seed, condition, spikes, events, morph_paths, pre_spike, fig_dir, min_events=5):
    """Plot dendritic voltage traces in `pre_spike` intervals for `gids` if events are detected in at least
    `min_events` dendritic sections"""
    for gid in tqdm(events["gid"].unique(), desc="Plotting traces for selected events"):
        gid_events = events.loc[(events["gid"] == gid) & (events["condition"] == condition)].sort_values("start")
        spike_times = spikes.loc[(spikes["gid"] == gid) & (spikes["condition"] == condition), "spike_times"].to_numpy()
        # count events before each spike (and only use the ones that are visible on enough dendritic sections)
        counts = gid_events.groupby(pd.cut(gid_events["start"], spike_times))["start"].count()
        counts = counts[counts >= min_events]
        if len(counts):
            pklf_name = os.path.join(results_dir, "%s_a%i_%s_voltages.pkl" % (seed, gid, condition))
            traces = pd.read_pickle(pklf_name)
            basal_df, apical_df = get_dendritic_path_lengths(morph_paths[gid])
            basal_range = [basal_df.iloc[0]["path_length"].astype(int), basal_df.iloc[-1]["path_length"].astype(int)]
            apical_range = [apical_df.iloc[0]["path_length"].astype(int), apical_df.iloc[-1]["path_length"].astype(int)]
            basal_traces = traces[basal_df["section"].to_numpy()]
            apical_traces = traces[apical_df["section"].to_numpy()]
            for w, _ in counts.items():
                plot_basal = basal_traces.loc[((w.right - pre_spike) < basal_traces.index)
                                              & (basal_traces.index < w.right), :].to_numpy().transpose()
                plot_apical = apical_traces.loc[((w.right - pre_spike) < apical_traces.index)
                                                & (apical_traces.index < w.right), :].to_numpy().transpose()
                fig_name = os.path.join(fig_dir, "a%i_%s_dendritic_traces_at%.1f.png" % (gid, condition, w.right))
                plot_dend_traces(plot_basal, plot_apical, basal_range, apical_range, [-pre_spike, 0], fig_name)


def main(config_path, threshold=-30, sustain_for=10, pre_spike=50):
    config = Config(config_path)
    sbatch_dir = os.path.join(config.root_path, "sbatch")
    results_dir = os.path.join(config.root_path, "analyses", "rerun_results")
    with open(os.path.join(sbatch_dir, "simulated_gids.pkl"), "rb") as f:
        simulated_gids = pickle.load(f)
    c = utils.get_bluepy_circuit_from_root_path(config.root_path)

    for seed, assembly_gids in simulated_gids.items():
        df, spikes = analyse_spikes(config, seed, assembly_gids)
        plot_across_conditions(df, "rate", os.path.join(config.fig_path, "spike_rate_across_conditions_%s.png" % seed))
        plot_across_conditions(df, "correlation", os.path.join(config.fig_path, "spike_corr_across_conditions_%s.png" % seed))

        events = analyse_traces(results_dir, seed, spikes, threshold, sustain_for, pre_spike)
        pklf_name = "%s_window_sustained_for%i_%ibefore_spikes.pkl" % (seed, sustain_for, pre_spike)
        events.to_pickle(os.path.join(results_dir, pklf_name))
        # plot events for L5 TTPCs in baseline conditions
        mtypes = utils.get_mtypes(c, df["gid"].unique())
        gids = mtypes[mtypes.isin(["L5_TPC:A", "L5_TPC:B"])].index.to_numpy()
        morph_paths = {gid: c.morph.get_filepath(gid) for gid in gids}
        fig_dir = os.path.join(config.fig_path, "%s_debug" % seed)
        utils.ensure_dir(fig_dir)
        plot_events(results_dir, seed, "baseline", spikes, events.loc[events["gid"].isin(gids)],
                    morph_paths, pre_spike, fig_dir)


if __name__ == "__main__":
    config_path = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/assemblyfire/configs/v7_10seeds_np.yaml"
    # write_launchscripts(config_path)
    main(config_path)

