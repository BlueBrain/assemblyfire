"""
Functionalities to investigate the relationship between synaptc clustering and dendritic nonlinearities
1st step: writes launch scripts that rerun selected cells in BGLibPy (see `assemblyfire/rerun_single_cell.py`)
          (with extra reporting and some modifications)
2nd step: load spiking data from sims. with modified conditions (and baseline), check rates, and
          if based on the new spike times the neuron would be still part of the assembly
last modified: András Ecker 01.2023
"""

import os
import h5py
import pickle
import numpy as np
import pandas as pd

import assemblyfire.utils as utils
from assemblyfire.config import Config
from assemblyfire.clustering import get_core_cell_idx
from assemblyfire.plots import plot_across_conditions


DSET_MEMBER = "member"
DSET_DEG = "degree"
DSET_CLST = "strength"
DSET_PVALUE = "pvalue"

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
    assembly_idx, all_gids = [], []
    spike_vectors = {"baseline": [], "passivedend": [], "noNMDA": []}
    for assembly_str, gids in assembly_gids.items():
        assembly_id = int(assembly_str.split("assembly")[1])
        for gid in gids:
            spikesf_name = os.path.join(results_dir, "%s_a%i_spikes.pkl" % (seed, gid))
            if os.path.isfile(spikesf_name):
                assembly_idx.append(assembly_id)
                all_gids.append(gid)
                spikes = pd.read_pickle(spikesf_name)
                for condition in ["baseline", "passivedend", "noNMDA"]:
                    spike_times = spikes.loc[spikes["condition"] == condition, "spike_times"].to_numpy()
                    binned_spikes, _ = np.histogram(spike_times, t_bins)
                    spike_vectors[condition].append(binned_spikes.reshape(1, -1))
            else:
                print("No spikes from: %s_assembly%i_a%i" % (seed, assembly_id, gid))
    spike_matrices = {condition: np.vstack(spike_vectors_) for condition, spike_vectors_ in spike_vectors.items()}
    return spike_matrices, np.array(all_gids), np.array(assembly_idx)


def analyse_results(config_path):
    config = Config(config_path)
    sbatch_dir = os.path.join(config.root_path, "sbatch")
    with open(os.path.join(sbatch_dir, "simulated_gids.pkl"), "rb") as f:
        simulated_gids = pickle.load(f)
    results_dir = os.path.join(config.root_path, "analyses", "rerun_results")
    df_columns = ["gid", "assembly", "rate", "correlation", "member", "condition"]

    t_bins = np.arange(config.t_start, config.t_end + config.bin_size, config.bin_size)
    rate_norm = (config.t_end - config.t_start) / 1e3  # ms -> s conversion
    clust_meta = utils.read_cluster_seq_data(config.h5f_name)

    for seed, assembly_gids in simulated_gids.items():
        # build spike matrices for checking assembly membership
        spike_matrices, gids, assembly_idx = _get_binned_spikes(results_dir, seed, assembly_gids, t_bins)
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
        # due to the random schuffled controls it can happen that neurons that were members of the assembly before,
        # won't be any longer, and we'll exclude those from the analysis...
        drop_gids = df.loc[(df["condition"] == "baseline") & (df["member"] == 0), "gid"].to_numpy()
        if len(drop_gids):
            df.drop(df.loc[df["gid"].isin(drop_gids)].index, inplace=True)
        # drop gids that don't spike in modified conditions (probably due to bad e-model)
        drop_gids = np.unique(df.loc[df["rate"] == 0, "gid"].to_numpy())
        if len(drop_gids):
            df.drop(df.loc[df["gid"].isin(drop_gids)].index, inplace=True)

        plot_across_conditions(df, "rate", os.path.join(config.fig_path, "spike_rate_across_conditions_%s.png" % seed))
        plot_across_conditions(df, "correlation", os.path.join(config.fig_path, "spike_corr_across_conditions_%s.png" % seed))


if __name__ == "__main__":
    config_path = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/assemblyfire/configs/v7_10seeds_np.yaml"
    # write_launchscripts(config_path)
    analyse_results(config_path)

