"""
Load assemblies from 2 HDF5 files and compare their overlap (intersection correlation)
last modified: András Ecker 09.2022
"""

import os

import assemblyfire.utils as utils
from assemblyfire.config import Config
from assemblyfire.plots import plot_assembly_intersection_corr


def _get_label(h5f_name):
    return os.path.split(h5f_name)[1].split('.')[0]


def assembly_overlaps(config1_path, config2_path):
    """Loads in assemblies and gets their intersection correlation (seed by seed)"""

    config1 = Config(config1_path)
    assembly_grp_dict1, _ = utils.load_assemblies_from_h5(config1.h5f_name, config1.h5_prefix_assemblies)
    config2 = Config(config2_path)
    assembly_grp_dict2, _ = utils.load_assemblies_from_h5(config2.h5f_name, config2.h5_prefix_assemblies)
    xlabel, ylabel = _get_label(config2.h5f_name), _get_label(config1.h5f_name)

    for seed, assembly_grp1 in assembly_grp_dict1.items():
        intersection_corrs = assembly_grp1.intersection_pattern_correlation(other=assembly_grp_dict2[seed])
        fig_name = os.path.join(config2.fig_path, "intersection_corr_%s.png" % seed)
        plot_assembly_intersection_corr(intersection_corrs, xlabel, ylabel, fig_name)


if __name__ == "__main__":
    config1_path = "../configs/v7_bbp-workflow.yaml"
    config2_path = "../configs/v7_bbp-workflow_L23.yaml"
    assembly_overlaps(config1_path, config2_path)
