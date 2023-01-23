"""
Load assemblies from (either from 2 HDF5 files, or from different prefixes) and compare them
last modified: András Ecker 01.2023
"""

import os
import numpy as np
from scipy.spatial.distance import cdist

import assemblyfire.utils as utils
from assemblyfire.config import Config
from assemblyfire.plots import plot_assembly_similarities, plot_consensus_vs_average_assembly_composition


def get_assembly_similarities(assembly_grp1, assembly_grp2):
    """Gets Jaccard distance of assembly neurons"""
    # creating ID matrices with the union of the gids
    gids1, gids2 = assembly_grp1.all, assembly_grp2.all
    gids = np.union1d(gids1, gids2)
    assembly_idx1 = np.zeros((len(assembly_grp1), len(gids)), dtype=int)
    assembly_idx1[:, np.in1d(gids, gids1, assume_unique=True)] = assembly_grp1.as_bool().transpose().astype(int)
    assembly_idx2 = np.zeros((len(assembly_grp2), len(gids)), dtype=int)
    assembly_idx2[:, np.in1d(gids, gids2, assume_unique=True)] = assembly_grp2.as_bool().transpose().astype(int)
    return 1 - cdist(assembly_idx1, assembly_idx2, "jaccard")


def _get_label(h5f_name):
    return os.path.split(h5f_name)[1].split('.')[0]


def assembly_similarities_from2configs(config1_path, config2_path):
    """Loads in assemblies and gets their Jaccard similarity (seed by seed)"""
    config1 = Config(config1_path)
    assembly_grp_dict1, _ = utils.load_assemblies_from_h5(config1.h5f_name, config1.h5_prefix_assemblies)
    config2 = Config(config2_path)
    assembly_grp_dict2, _ = utils.load_assemblies_from_h5(config2.h5f_name, config2.h5_prefix_assemblies)
    xlabel, ylabel = _get_label(config2.h5f_name), _get_label(config1.h5f_name)
    for seed, assembly_grp1 in assembly_grp_dict1.items():
        similarities = get_assembly_similarities(assembly_grp1, assembly_grp_dict2[seed])
        fig_name = os.path.join(config2.fig_path, "assembly_similarities_%s.png" % seed)
        plot_assembly_similarities(similarities, xlabel, ylabel, fig_name)


def consensus_at_fraction_thresholds(consensus_assembly_grp, frac_ths):
    """Create assembly groups (as those are used in this script) at different thresholds, where the thresholded variable
    is number of times contained AKA. `raw` method in `assemblyfire.assemblies()` over the number of instantiations
    TODO: maybe add this method to the `ConsensusAssembly` class"""
    from assemblyfire.assemblies import Assembly, AssemblyGroup
    all_gids = consensus_assembly_grp.all
    assembly_grps = {}
    for frac_th in frac_ths:
        assembly_lst = []
        for assembly in consensus_assembly_grp.assemblies:
            # `fracs` could be stored instead of recalculated again for all thresholds but whatever...
            fracs = assembly.__number_of_times_contained__() / len(assembly.instantiations)
            assembly_lst.append(Assembly(assembly.union.gids[fracs > frac_th], index=(assembly.idx[0], frac_th)))
        assembly_grps[frac_th] = AssemblyGroup(assembly_lst, all_gids)
    return assembly_grps


def consensus_vs_average_assembly_similarity(config_path, frac_ths):
    """Loads in consensus and average assemblies and gets their Jaccard similarity"""
    config = Config(config_path)
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_avg_assemblies)
    avg_assembly_grp = assembly_grp_dict["seed_average"]
    consensus_assemblies = utils.load_consensus_assemblies_from_h5(config.h5f_name, config.h5_prefix_consensus_assemblies)
    consensus_assembly_grp = utils.consensus_dict2assembly_grp(consensus_assemblies)

    # core (as defined bassed on p-value threshold vs. average assembly)
    similarities = get_assembly_similarities(avg_assembly_grp, consensus_assembly_grp)
    fig_name = os.path.join(config.fig_path, "consensus_vs_average_assembly_similarities.png")
    plot_assembly_similarities(similarities, "Consensus (core) assembly", "Average assembly", fig_name)

    # different thresholds, but not on p-value, but on franction of inst. contained instead
    assembly_grps = consensus_at_fraction_thresholds(consensus_assembly_grp, frac_ths)
    for frac_th, assembly_grp in assembly_grps.items():
        similarities = get_assembly_similarities(avg_assembly_grp, assembly_grp)
        fig_name = os.path.join(config.fig_path, "consensus_@frac%.1f_vs_average_assembly_similarities.png" % frac_th)
        plot_assembly_similarities(similarities, "Consensus @frac%.1f assembly" % frac_th, "Average assembly", fig_name)


def consensus_vs_average_assembly_composition(config_path, avg_assembly_id, consensus_assembly_id):
    """TODO"""
    config = Config(config_path)
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_avg_assemblies)
    avg_assembly_grp = assembly_grp_dict["seed_average"]
    avg_gids = avg_assembly_grp.loc(avg_assembly_id).gids
    consensus_assemblies = utils.load_consensus_assemblies_from_h5(config.h5f_name, config.h5_prefix_consensus_assemblies)
    consensus_assembly = consensus_assemblies["cluster%i" % consensus_assembly_id]

    gids, ns = consensus_assembly.union.gids, consensus_assembly.__number_of_times_contained__()
    gids_at_n = [gids[ns == n] for n in np.unique(ns)]
    intersection_at_n = [len(np.intersect1d(gids, avg_gids, assume_unique=True)) for gids in gids_at_n]
    diff_at_n = [len(np.setdiff1d(gids, avg_gids, assume_unique=True)) for gids in gids_at_n]
    fig_name = os.path.join(config.fig_path, "consenses%i_vs_average%i.png" % (consensus_assembly_id, avg_assembly_id))
    plot_consensus_vs_average_assembly_composition(intersection_at_n, diff_at_n, fig_name)


if __name__ == "__main__":
    # config1_path = "../configs/v7_bbp-workflow.yaml"
    # config2_path = "../configs/v7_bbp-workflow_L23.yaml"
    # assembly_similarities_from2configs(config1_path, config2_path)
    config_path = "../configs/v7_10seeds_np.yaml"
    consensus_vs_average_assembly_similarity(config_path, frac_ths=[0.2, 0.4, 0.6, 0.8])
    consensus_vs_average_assembly_composition(config_path, 7, 1)

