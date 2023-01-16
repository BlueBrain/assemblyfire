"""
Correlate consensus assembly membership and single cell features
last modified: András Ecker 01.2023
"""

import os
import numpy as np
import pandas as pd

from assemblyfire.config import Config
from assemblyfire.utils import load_single_cell_features_from_h5, load_consensus_assemblies_from_h5, get_sim_path
from assemblyfire.plots import plot_consensus_r_spikes, plot_coreness_r_spike


def consensus_vs_single_cell_features(config_path):
    """Loads in consensus assemblies and single cell features from h5 file
    and plots the distributions of spike time reliability for consensus assemblies"""
    config = Config(config_path)

    # loading single cell features and consensus assemblies from h5
    single_cell_features, _ = load_single_cell_features_from_h5(config.h5f_name, prefix=config.h5_prefix_single_cell)
    consensus_assemblies = load_consensus_assemblies_from_h5(config.h5f_name,
                                                             prefix=config.h5_prefix_consensus_assemblies)
    all_gids, r_spikes = single_cell_features["gids"], single_cell_features["r_spikes"]

    dfs, cons_assembly_gids = [], []
    for cons_assembly_id, assembly in consensus_assemblies.items():
        gids = assembly.gids
        df = pd.DataFrame(data=r_spikes[np.in1d(all_gids, gids)], index=gids, columns=["r_spike"], dtype=np.float32)
        df["consensus assembly id"] = cons_assembly_id.split("cluster")[1]
        dfs.append(df)
        cons_assembly_gids.extend(gids)
    gids = np.unique(cons_assembly_gids)
    idx = ~np.in1d(all_gids, gids)
    df = pd.DataFrame(data=r_spikes[idx], index=all_gids[idx], columns=["r_spike"], dtype=np.float32)
    df["consensus assembly id"] = "non assembly"
    dfs.append(df)
    df = pd.concat(dfs, join="inner")  # inner join will keep gids that are part of multiple assemblies
    plot_consensus_r_spikes(df, os.path.join(config.fig_path, "consensus_r_spikes.png"))

    '''
    cons_assembly_idx = np.sort(np.array([int(cons_assembly_id.split("cluster")[1])  # order them for plotting...
                                          for cons_assembly_id in list(consensus_assemblies.keys())]))
    coreness, union_r_spikes = [], []
    for cons_assembly_id in cons_assembly_idx:
        assembly = consensus_assemblies["cluster%i" % cons_assembly_id]
        coreness.append(assembly.coreness)
        union_r_spikes.append(r_spikes[np.in1d(all_gids, assembly.union.gids)])
    plot_coreness_r_spike(union_r_spikes, coreness, os.path.join(config.fig_path, "coreness_r_spikes.png"))
    '''


if __name__ == "__main__":
    consensus_vs_single_cell_features("../configs/v7_10seeds_np.yaml")
