"""
In degrees, simplex counts and pot/dep ratios of assemblies and consensus assemblies
last modified: András Ecker 01.2022
"""

import os
from tqdm import tqdm

import assemblyfire.utils as utils
from assemblyfire.config import Config
from assemblyfire.topology import AssemblyTopology, in_degree_assemblies, simplex_counts_assemblies,\
                                  simplex_counts_consensus_instantiations
from assemblyfire.plots import plot_efficacy, plot_in_degrees, plot_simplex_counts, plot_simplex_counts_consensus


def assembly_efficacy(config_path):
    """Loads in assemblies and counts synapses initialized at depressed (rho=0) and potentiated (rho=1) states"""

    config = Config(config_path)
    sim_paths = utils.get_sim_path(config.root_path)
    c = utils.get_bluepy_circuit(sim_paths.iloc[0])
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)

    for seed, assembly_grp in assembly_grp_dict.items():
        efficacies = {assembly.idx: utils.get_syn_properties(c,
                      utils.get_syn_idx(c, assembly.gids, assembly.gids), ["rho0_GB"])["rho0_GB"].value_counts()
                      for assembly in tqdm(assembly_grp.assemblies, desc="%s assembly efficacies" % seed)}
        fig_name = os.path.join(config.fig_path, "efficacy_%s.png" % seed)
        plot_efficacy(efficacies, fig_name)


def assembly_in_degree(config_path):
    """Loads in assemblies and plots in degrees (seed by seed)"""

    config = Config(config_path)
    conn_mat = AssemblyTopology.from_h5(config.h5f_name,
                                        prefix=config.h5_prefix_connectivity, group_name="full_matrix")
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)
    in_degrees, in_degrees_control = in_degree_assemblies(assembly_grp_dict, conn_mat)
    for seed, in_degree in in_degrees.items():
        fig_name = os.path.join(config.fig_path, "in_degrees_%s.png" % seed)
        plot_in_degrees(in_degree, in_degrees_control[seed], fig_name)


def assembly_simplex_counts(config_path):
    """Loads in assemblies and plots simplex counts (seed by seed and then for consensus assemblies)"""

    config = Config(config_path)
    conn_mat = AssemblyTopology.from_h5(config.h5f_name,
                                        prefix=config.h5_prefix_connectivity, group_name="full_matrix")
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)
    consensus_assembly_dict = utils.load_consensus_assemblies_from_h5(config.h5f_name,
                                                                      prefix=config.h5_prefix_consensus_assemblies)
    simplex_counts, simplex_counts_control = simplex_counts_assemblies(assembly_grp_dict, conn_mat)
    for seed, simplices in simplex_counts.items():
        fig_name = os.path.join(config.fig_path, "simplex_counts_%s.png" % seed)
        plot_simplex_counts(simplices, simplex_counts_control[seed], fig_name)

    simplex_counts, simplex_counts_control = simplex_counts_consensus_instantiations(consensus_assembly_dict, conn_mat)
    fig_name = os.path.join(config.fig_path, "simplex_counts_consensus.png")
    plot_simplex_counts_consensus(simplex_counts, simplex_counts_control, fig_name)


if __name__ == "__main__":
    config_path = "../configs/v7_bbp-workflow.yaml"
    assembly_efficacy(config_path)
    # assembly_in_degree(config_path)
    # assembly_simplex_counts(config_path)
