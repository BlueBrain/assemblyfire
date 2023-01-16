"""
Load assemblies from (either from 2 HDF5 files, or from different prefixes) and compare them
last modified: András Ecker 01.2023
"""

import os
import numpy as np
from scipy.spatial.distance import cdist

import assemblyfire.utils as utils
from assemblyfire.config import Config
from assemblyfire.plots import plot_assembly_similarities


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


def _create_assembly_grp_from_consensus_core(consensus_assemblies):
    """Create AssemblyGroup object (the format used in `get_assembly_similarities()`)
    from the core's of consensus assemblies"""
    from assemblyfire.assemblies import Assembly, AssemblyGroup
    consensus_assembly_idx = np.sort([int(key.split("cluster")[1]) for key in list(consensus_assemblies.keys())])
    all_gids, assembly_lst = [], []
    for consensus_assembly_id in consensus_assembly_idx:
        gids = consensus_assemblies["cluster%i" % consensus_assembly_id].gids
        all_gids.extend(gids)
        assembly_lst.append(Assembly(gids, index=consensus_assembly_id))
    return AssemblyGroup(assemblies=assembly_lst, all_gids=np.unique(all_gids))


def consensus_vs_average_assembly_similarity(config_path):
    """Loads in consensus and average assemblies and gets their Jaccard similarity"""
    config = Config(config_path)
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_avg_assemblies)
    avg_assembly_grp = assembly_grp_dict["seed_average"]
    consensus_assemblies = utils.load_consensus_assemblies_from_h5(config.h5f_name, config.h5_prefix_consensus_assemblies)
    consensus_assembly_grp = _create_assembly_grp_from_consensus_core(consensus_assemblies)
    similarities = get_assembly_similarities(avg_assembly_grp, consensus_assembly_grp)
    fig_name = os.path.join(config.fig_path, "consensus_vs_average_assembly_similarities.png")
    plot_assembly_similarities(similarities, "Consensus assembly", "Average assembly", fig_name)


if __name__ == "__main__":
    # config1_path = "../configs/v7_bbp-workflow.yaml"
    # config2_path = "../configs/v7_bbp-workflow_L23.yaml"
    # assembly_similarities_from2configs(config1_path, config2_path)
    config_path = "../configs/v7_10seeds_np.yaml"
    consensus_vs_average_assembly_similarity(config_path)

