"""
In degrees, simplex counts, pot/dep ratios, and membership probabilities of assemblies
last modified: András Ecker 10.2022
"""

import os
from tqdm import tqdm
import numpy as np
import pandas as pd

from assemblyfire.config import Config
import assemblyfire.utils as utils
import assemblyfire.plots as plots
from assemblyfire.topology import AssemblyTopology, in_degree_assemblies, simplex_counts_assemblies


def assembly_efficacy(config):
    """Loads in assemblies and plots synapses initialized at depressed (rho=0) and potentiated (rho=1) states"""

    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)
    c = utils.get_bluepy_circuit(utils.get_sim_path(config.root_path).iloc[0])
    rhos = utils.get_rho0s(c, config.target)  # get all rhos in one go and then index them as needed

    for seed, assembly_grp in tqdm(assembly_grp_dict.items(), desc="Getting efficacies"):
        efficacies = {assembly.idx[0]: rhos.loc[rhos["pre_gid"].isin(assembly.gids)
                                                & rhos["post_gid"].isin(assembly.gids), "rho"].value_counts()
                      for assembly in assembly_grp.assemblies}
        fig_name = os.path.join(config.fig_path, "efficacy_%s.png" % seed)
        plots.plot_efficacy(efficacies, fig_name)


def assembly_in_degree(config):
    """Loads in assemblies and plots in degrees within the assemblies and in their control models (seed by seed)"""

    conn_mat = AssemblyTopology.from_h5(config.h5f_name,
                                        prefix=config.h5_prefix_connectivity, group_name="full_matrix")
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)

    in_degrees, in_degrees_control = in_degree_assemblies(assembly_grp_dict, conn_mat)
    for seed, in_degree in in_degrees.items():
        fig_name = os.path.join(config.fig_path, "in_degrees_%s.png" % seed)
        plots.plot_in_degrees(in_degree, in_degrees_control[seed], fig_name)

    '''
    in_degrees, in_degrees_control = in_degree_assemblies(assembly_grp_dict, conn_mat, post_id=0)
    for seed, in_degree in in_degrees.items():
        fig_name = os.path.join(config.fig_path, "cross_assembly_in_degrees_%s.png" % seed)
        plots.plot_in_degrees(in_degree, in_degrees_control[seed], fig_name, xlabel="Cross assembly (any to 0) in degree")
    '''


def assembly_simplex_counts(config):
    """Loads in assemblies and plots simplex counts in assemblies and control models (seed by seed)"""

    conn_mat = AssemblyTopology.from_h5(config.h5f_name,
                                        prefix=config.h5_prefix_connectivity, group_name="full_matrix")
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)

    simplex_counts, simplex_counts_control = simplex_counts_assemblies(assembly_grp_dict, conn_mat)
    for seed, simplices in simplex_counts.items():
        fig_name = os.path.join(config.fig_path, "simplex_counts_%s.png" % seed)
        plots.plot_simplex_counts(simplices, simplex_counts_control[seed], fig_name)


def _bin_gids_by_innervation(indegree_dict, gids, min_samples):
    """Creates lookups of gids in optimal bins for each pattern
    (optimal bins are determined based on their innervation profile)"""
    binned_gids, bin_centers_dict = {key: {} for key in list(indegree_dict.keys())}, {}
    for key, indegrees in indegree_dict.items():
        idx = np.where(indegrees > 0.)[0]  # sometimes -1s are used as placeholders...
        gids_tmp, indegrees = gids[idx], indegrees[idx]
        bin_edges, bin_centers = utils.determine_bins(*np.unique(indegrees, return_counts=True), min_samples)
        bin_centers_dict[key] = bin_centers
        bin_idx = np.digitize(indegrees, bin_edges, right=True)
        for i, center in enumerate(bin_centers):
            binned_gids[key][center] = gids_tmp[bin_idx == i+1]
    return binned_gids, bin_centers_dict


def _mi_implementation(degree_counts, degree_p):
    """
    Analyzes how much of the uncertainty of assembly membership is explained away when one considers the strengths
    of innervation from a given pre-synaptic target (in terms of in-degree).
    :param degree_counts: The number of neurons in the simulation that have a given degree. One entry per degree-bin.
    :param degree_p: The probability that neurons with a given degree are members of the assembly in question.
                     (Must have same length as degree_counts.) Note: Yes, for this analysis the actual _value_
                     of a degree-bin (which min/max degree does it represent?) is irrelevant. - Michael W.R.
    :return: membership_entropy: The prior entropy of assembly membership.
    :return: posterior_entropy: The posterior entropy of assembly membership conditional on the innervation degree.
    """

    def entropy(p):
        return -np.log2(p) * p - np.log2(1 - p) * (1 - p)

    def entropy_vec(p_vec):
        return np.nansum(np.vstack([-np.log2(p_vec) * p_vec, -np.log2(1 - p_vec) * (1 - p_vec)]), axis=0)

    degree_counts, degree_p = np.array(degree_counts), np.array(degree_p)
    overall_p = (degree_counts * degree_p).sum() / degree_counts.sum()
    membership_entropy = entropy(overall_p)
    posterior_entropy = (entropy_vec(degree_p) * degree_counts).sum() / degree_counts.sum()
    return membership_entropy, posterior_entropy


def _sign_of_correlation(degree_vals, degree_p):
    """
    Analyzes whether the strength of innervation from a given pre-synaptic target (in terms of in-degree) is rather
    increasing (positive sign) or decreasing (negative sign) the probability that the innervated neuron is member of
    an assembly.
    :param degree_vals: The possible values of degrees for the innervated neurons. e.g. the centers of degree-bins.
    :param degree_p: The probability that neurons with a given degree are members of the assembly in question.
                     (Must have same length as degree_counts.)
    :return: sign: -1 if stronger innervation decreases probability of membership; 1 if it rather increases it
    """
    degree_vals, degree_p = np.array(degree_vals), np.array(degree_p)
    idx = np.argsort(degree_vals)
    return np.sign(np.polyfit(degree_vals[idx], degree_p[idx], 1)[0])


def frac_entropy_explained_by_indegree(config, min_samples=100):
    """Loads in assemblies and for each of them plots the probabilities of assembly membership
    vs. in degree (from the assembly neurons) as well as the (relative) loss in entropy. i.e. How much percent
    of the uncertainty (in assembly membership) can be explained by pure structural innervation"""

    conn_mat = AssemblyTopology.from_h5(config.h5f_name,
                                        prefix=config.h5_prefix_connectivity, group_name="full_matrix")
    gids = conn_mat.gids
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)

    assembly_indegrees_dict = {}
    for seed, assembly_grp in assembly_grp_dict.items():
        assembly_indegrees = {assembly.idx[0]: conn_mat.degree(assembly.gids, gids)
                              for assembly in assembly_grp.assemblies}
        assembly_indegrees_dict[seed] = pd.DataFrame(assembly_indegrees, index=gids)
        binned_gids, bin_centers = _bin_gids_by_innervation(assembly_indegrees, gids, min_samples)

        chance_levels = {}
        bin_centers_plot = {pre_assembly: {} for pre_assembly in list(binned_gids.keys())}
        assembly_probs = {pre_assembly: {} for pre_assembly in list(binned_gids.keys())}
        assembly_mi = {pre_assembly: {} for pre_assembly in list(binned_gids.keys())}
        for assembly in assembly_grp.assemblies:
            assembly_id = assembly.idx[0]
            idx = np.in1d(gids, assembly.gids, assume_unique=True)
            chance_levels[assembly_id] = idx.sum() / len(idx)
            for pre_assembly, binned_gids_tmp in binned_gids.items():
                probs, counts, vals = [], [], []
                for bin_center in bin_centers[pre_assembly]:
                    idx = np.in1d(binned_gids_tmp[bin_center], assembly.gids, assume_unique=True)
                    probs.append(idx.sum() / len(idx))
                    counts.append(len(binned_gids_tmp[bin_center]))
                    vals.append(bin_center)
                bin_centers_plot[pre_assembly][assembly_id] = bin_centers[pre_assembly]
                assembly_probs[pre_assembly][assembly_id] = np.array(probs)
                me, pe = _mi_implementation(counts, probs)
                assembly_mi[pre_assembly][assembly_id] = _sign_of_correlation(vals, probs) * (1.0 - pe / me)

        fig_name = os.path.join(config.fig_path, "assembly_prob_from_indegree_%s.png" % seed)
        palette = {assembly.idx[0]: "pre_assembly_color" for assembly in assembly_grp.assemblies}
        plots.plot_assembly_prob_from(bin_centers_plot, assembly_probs, chance_levels,
                                      "In degree", palette, fig_name)
        fig_name = os.path.join(config.fig_path, "frac_entropy_explained_by_recurrent_innervation_%s.png" % seed)
        plots.plot_frac_entropy_explained_by(pd.DataFrame(assembly_mi).transpose(), "Innervation by assembly", fig_name)

    return assembly_indegrees_dict


def _nnd_df_to_dict(nnd_df):
    """Converts DataFrame from `clustering.syn_nearest_neighbour_distances()` to the dict format
    which is compatible with the `_bin_gids_by_innervation()` helper functions above"""
    gids = nnd_df.index.to_numpy()
    assembly_idx = nnd_df.columns.to_numpy()
    return {assembly_id: nnd_df[assembly_id].to_numpy() for assembly_id in assembly_idx}, gids


def frac_entropy_explained_by_syn_nnd(config, min_samples=100):
    """Loads in assemblies and for each (sub)target neurons calculates the (normalized) nearest neighbour distance
    for assembly synapses (which is meant to be a parameter free measure of synapse clustering) and plots the prob.
    of assembly membership vs. this measure"""
    from assemblyfire.clustering import syn_nearest_neighbour_distances

    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)
    c = utils.get_bluepy_circuit(utils.get_sim_path(config.root_path).iloc[0])
    loc_df = utils.get_loc_df(config.syn_clustering_lookup_df_pklfname, c, config.target, config.syn_clustering_target)
    mtypes = utils.get_mtypes(c, utils.get_gids(c, config.target)).reset_index()
    mtypes.rename(columns={"index": "gid"}, inplace=True)

    assembly_nnds_dict = {}
    for seed, assembly_grp in assembly_grp_dict.items():
        ctrl_assembly_grp = assembly_grp.random_categorical_controls(mtypes, "mtype")
        assembly_nnds, ctrl_nnds = syn_nearest_neighbour_distances(loc_df, assembly_grp, ctrl_assembly_grp)
        assembly_nnds_dict[seed] = assembly_nnds
        binned_gids, bin_centers = _bin_gids_by_innervation(*_nnd_df_to_dict(assembly_nnds), min_samples)

        chance_levels = {}
        bin_centers_plot = {pre_assembly: {} for pre_assembly in list(binned_gids.keys())}
        assembly_probs = {pre_assembly: {} for pre_assembly in list(binned_gids.keys())}
        assembly_mi = {pre_assembly: {} for pre_assembly in list(binned_gids.keys())}
        for assembly in assembly_grp.assemblies:
            assembly_id = assembly.idx[0]
            idx = np.in1d(assembly_nnds[assembly_id].to_numpy(), assembly.gids, assume_unique=True)
            chance_levels[assembly_id] = idx.sum() / len(idx)
            for pre_assembly, binned_gids_tmp in binned_gids.items():
                probs, counts, vals = [], [], []
                for bin_center in bin_centers[pre_assembly]:
                    idx = np.in1d(binned_gids_tmp[bin_center], assembly.gids, assume_unique=True)
                    probs.append(idx.sum() / len(idx))
                    counts.append(len(binned_gids_tmp[bin_center]))
                    vals.append(bin_center)
                bin_centers_plot[pre_assembly][assembly_id] = bin_centers[pre_assembly]
                assembly_probs[pre_assembly][assembly_id] = np.array(probs)
                me, pe = _mi_implementation(counts, probs)
                assembly_mi[pre_assembly][assembly_id] = _sign_of_correlation(vals, probs) * (1.0 - pe / me)

        fig_name = os.path.join(config.fig_path, "assembly_prob_from_syn_nearest_neighbour_%s.png" % seed)
        palette = {assembly.idx[0]: "pre_assembly_color" for assembly in assembly_grp.assemblies}
        plots.plot_assembly_prob_from(bin_centers_plot, assembly_probs, chance_levels,
                                      "Synapse nearest neighbour distance", palette, fig_name)
        fig_name = os.path.join(config.fig_path, "frac_entropy_explained_by_syn_nearest_neighbour_%s.png" % seed)
        plots.plot_frac_entropy_explained_by(pd.DataFrame(assembly_mi).transpose(),
                                             "Synapse nearest neighbour from assembly", fig_name)

    return assembly_nnds_dict


def assembly_prob_from_indegree_and_syn_nnd(config, assembly_indegrees_dict, assembly_nnds_dict,
                                            palette, min_samples=100):
    """Combines previous results and weights indegrees with synapse neighbour distances
    (and then predicts assembly membership from that for all assemblies)"""

    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)
    keys = list(palette.keys())

    for seed, assembly_grp in assembly_grp_dict.items():
        assembly_indegrees, assembly_nnds = assembly_indegrees_dict[seed], assembly_nnds_dict[seed]
        chance_levels = {}
        assembly_probs = {key: {} for key in keys}
        bin_centers_dict = {key: {} for key in keys}
        for assembly in assembly_grp.assemblies:
            assembly_id = assembly.idx[0]
            binned_assembly_nnds = pd.qcut(assembly_nnds.loc[assembly_nnds[assembly_id] > 0, assembly_id],
                                           len(keys), labels=keys)
            idx = np.in1d(binned_assembly_nnds.index.to_numpy(), assembly.gids, assume_unique=True)
            chance_levels[assembly_id] = idx.sum() / len(idx)
            for key in keys:
                gids = binned_assembly_nnds.loc[binned_assembly_nnds == key].index.to_numpy()
                key_assembly_indegrees = assembly_indegrees.loc[gids, assembly_id]
                bin_edges, bin_centers = utils.determine_bins(*np.unique(key_assembly_indegrees.to_numpy(),
                                                                         return_counts=True), min_samples)
                bin_centers_dict[key][assembly_id] = bin_centers
                bin_idx = np.digitize(key_assembly_indegrees.to_numpy(), bin_edges, right=True)
                gids_tmp, probs = key_assembly_indegrees.index.to_numpy(), []
                for i, center in enumerate(bin_centers):
                    idx = np.in1d(gids_tmp[bin_idx == i + 1], assembly.gids, assume_unique=True)
                    probs.append(idx.sum() / len(idx))
                assembly_probs[key][assembly_id] = np.array(probs)

        fig_name = os.path.join(config.fig_path, "assembly_prob_from_indegree_syn_nnd_%s.png" % seed)
        plots.plot_assembly_prob_from(bin_centers_dict, assembly_probs, chance_levels,
                                      "Innervation by assembly (weighted by synapse nnd.)", palette, fig_name)


def assembly_prob_from_sinks(config, palette, min_samples=100):
    """Loads in assemblies and plots generalized in degrees (sinks of high dim. simplices) within the assemblies
    (seed by seed). Simplices are found in a way that all non-sink neurons are guaranteed to be within the assembly"""

    conn_mat = AssemblyTopology.from_h5(config.h5f_name,
                                        prefix=config.h5_prefix_connectivity, group_name="full_matrix")
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)
    gids = conn_mat.gids
    dims = list(palette.keys())

    for seed, assembly_grp in assembly_grp_dict.items():
        chance_levels, bin_centers_dict, assembly_probs = {}, {dim: {} for dim in dims}, {dim: {} for dim in dims}
        for assembly in tqdm(assembly_grp.assemblies, desc="Iterating over assemblies"):
            idx = np.in1d(gids, assembly.gids, assume_unique=True)
            chance_levels[assembly.idx[0]] = idx.sum() / len(idx)
            simplex_list = conn_mat.simplex_list(assembly.gids, gids)
            for dim in dims:
                sink_counts, _ = np.histogram(simplex_list[dim][:, -1], np.arange(len(gids) + 1))
                bin_edges, bin_centers = utils.determine_bins(*np.unique(sink_counts, return_counts=True), min_samples)
                bin_edges, bin_centers = np.insert(bin_edges, 0, -1), np.insert(bin_centers, 0, 0)
                bin_centers_dict[dim][assembly.idx[0]] = bin_centers
                bin_idx = np.digitize(sink_counts, bin_edges, right=True)
                probs = []
                for i, center in enumerate(bin_centers):
                    idx = np.in1d(gids[bin_idx == i + 1], assembly.gids, assume_unique=True)
                    probs.append(idx.sum() / len(idx))
                assembly_probs[dim][assembly.idx[0]] = np.array(probs)

        fig_name = os.path.join(config.fig_path, "assembly_prob_from_simplex_dim_%s.png" % seed)
        plots.plot_assembly_prob_from(bin_centers_dict, assembly_probs, chance_levels,
                                      "Generalized in degree (#simplex sinks)", palette, fig_name, True)


def get_pattern_innervation(config):
    """Looks up for each neuron how many pattern fibers innervate it"""
    from conntility.circuit_models import circuit_connection_matrix

    # get (sparse) connectivity matrix between the input fibers and neurons in the circuit
    c = utils.get_bluepy_circuit(utils.get_sim_path(config.root_path).iloc[0])
    post_gids = utils.get_gids(c, config.target)
    pattern_gids = utils.get_pattern_gids(config.pattern_gids_fname)
    pre_gids = np.unique(np.concatenate([gids for _, gids in pattern_gids.items()]))
    input_conn_mat = circuit_connection_matrix(c, config.patterns_projection_name, pre_gids, post_gids).tocsr()
    # for each neurons (and for each patterns) get how many pattern fibers innervate it
    pattern_indegrees = {}
    for pattern_name, gids in pattern_gids.items():
        pattern_idx = np.in1d(pre_gids, gids, assume_unique=True)
        pattern_indegrees[pattern_name] = np.array(input_conn_mat[pattern_idx].sum(axis=0)).flatten()
    return pattern_indegrees, post_gids


def frac_entropy_explained_by_patterns(config, min_samples=100):
    """Loads in assemblies and for each of them plots the probabilities of assembly membership
    vs. purely structural innervation by the input patterns as well as the (relative) loss in entropy i.e. How much
    percent of the uncertainty (in assembly membership) can be explained by pure structural innervation from VPM"""

    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)
    pattern_indegrees, gids = get_pattern_innervation(config)
    binned_gids, bin_centers = _bin_gids_by_innervation(pattern_indegrees, gids, min_samples)

    for seed, assembly_grp in assembly_grp_dict.items():
        chance_levels = {}
        bin_centers_plot = {pattern_name: {} for pattern_name in list(pattern_indegrees.keys())}
        assembly_probs = {pattern_name: {} for pattern_name in list(pattern_indegrees.keys())}
        assembly_mi = {pattern_name: {} for pattern_name in list(pattern_indegrees.keys())}
        for assembly in assembly_grp.assemblies:
            assembly_id = assembly.idx[0]
            idx = np.in1d(gids, assembly.gids, assume_unique=True)
            chance_levels[assembly_id] = idx.sum() / len(idx)
            for pattern_name, binned_gids_tmp in binned_gids.items():
                probs, counts, vals = [], [], []
                for bin_center in bin_centers[pattern_name]:
                    idx = np.in1d(binned_gids_tmp[bin_center], assembly.gids, assume_unique=True)
                    probs.append(idx.sum() / len(idx))
                    counts.append(len(binned_gids_tmp[bin_center]))
                    vals.append(bin_center)
                bin_centers_plot[pattern_name][assembly_id] = bin_centers[pattern_name]
                assembly_probs[pattern_name][assembly_id] = np.array(probs)
                me, pe = _mi_implementation(counts, probs)
                assembly_mi[pattern_name][assembly_id] = (1.0 - pe / me) * _sign_of_correlation(vals, probs)

        fig_name = os.path.join(config.fig_path, "assembly_prob_from_patterns_%s.png" % seed)
        plots.plot_assembly_prob_from(chance_levels, bin_centers_plot, assembly_probs,
                                      "In degree from patterns", "patterns", fig_name)
        fig_name = os.path.join(config.fig_path, "frac_entropy_explained_by_patterns_%s.png" % seed)
        plots.plot_frac_entropy_explained_by(pd.DataFrame(assembly_mi).transpose(), "Innervation by pattern", fig_name)


if __name__ == "__main__":
    config = Config("../configs/v7_bbp-workflow.yaml")
    # assembly_efficacy(config)
    # assembly_in_degree(config)
    # assembly_simplex_counts(config)
    # assembly_indegrees = frac_entropy_explained_by_indegree(config)
    # assembly_nnds = frac_entropy_explained_by_syn_nnd(config)
    # assembly_prob_from_indegree_and_syn_nnd(config, assembly_indegrees, assembly_nnds,
    #                                         {"below avg.": "assembly_color", "avg.": "gray", "above avg.": "black"})
    assembly_prob_from_sinks(config, {2: "lightgray", 3: "gray", 4: "black", 5: "assembly_color"})
    # frac_entropy_explained_by_patterns(config)

