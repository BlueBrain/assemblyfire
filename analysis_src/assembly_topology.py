"""
In degrees, simplex counts, and pot/dep ratios of assemblies
last modified: András Ecker 09.2022
"""

import os
from tqdm import tqdm
import numpy as np
import pandas as pd

import assemblyfire.utils as utils
import assemblyfire.plots as plots
from assemblyfire.config import Config
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
    """Loads in assemblies and plots in degrees within assemblies and cross assemblies seed-by-seed
    (for cross assembly in degrees the postsynaptic target assembly is fixed: the late assembly)"""

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
    """Loads in assemblies and plots simplex counts (seed by seed
    and then for all instantiations per consensus assemblies)"""

    conn_mat = AssemblyTopology.from_h5(config.h5f_name,
                                        prefix=config.h5_prefix_connectivity, group_name="full_matrix")
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)

    simplex_counts, simplex_counts_control = simplex_counts_assemblies(assembly_grp_dict, conn_mat)
    for seed, simplices in simplex_counts.items():
        fig_name = os.path.join(config.fig_path, "simplex_counts_%s.png" % seed)
        plots.plot_simplex_counts(simplices, simplex_counts_control[seed], fig_name)


def _bin_gids_by_innervation(nconns_dict, gids, min_samples):
    """Creates lookups of gids in optimal bins for each pattern
    (optimal bins are determined based on their innervation profile)"""
    binned_gids, bin_centers_dict = {key: {} for key in list(nconns_dict.keys())}, {}
    for key, nconns in nconns_dict.items():
        idx = np.where(nconns > 0.)[0]  # sometimes -1s are used as placeholders...
        gids_tmp, nconns = gids[idx], nconns[idx]
        bin_edges, bin_centers = utils.determine_bins(*np.unique(nconns, return_counts=True), min_samples)
        bin_centers_dict[key] = bin_centers
        bin_idx = np.digitize(nconns, bin_edges, right=True)
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

        chance_levels, assembly_probs = {}, {}
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
                if pre_assembly == assembly_id:
                    assembly_probs[assembly_id] = np.array(probs)
                me, pe = _mi_implementation(counts, probs)
                assembly_mi[pre_assembly][assembly_id] = _sign_of_correlation(vals, probs) * (1.0 - pe / me)

        fig_name = os.path.join(config.fig_path, "assembly_prob_from_indegree_%s.png" % seed)
        plots.plot_assembly_prob_from(bin_centers, assembly_probs, chance_levels, "In degree", fig_name)
        fig_name = os.path.join(config.fig_path, "frac_entropy_explained_by_recurrent_innervation_%s.png" % seed)
        plots.plot_frac_entropy_explained_by(pd.DataFrame(assembly_mi), "Innervation by assembly", fig_name)

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

        chance_levels, assembly_probs = {}, {}
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
                if pre_assembly == assembly_id:
                    assembly_probs[assembly_id] = np.array(probs)
                me, pe = _mi_implementation(counts, probs)
                assembly_mi[pre_assembly][assembly_id] = _sign_of_correlation(vals, probs) * (1.0 - pe / me)

        fig_name = os.path.join(config.fig_path, "assembly_prob_from_syn_nearest_neighbour_%s.png" % seed)
        plots.plot_assembly_prob_from(bin_centers, assembly_probs, chance_levels,
                                      "Synapse nearest neighbour distance", fig_name)
        fig_name = os.path.join(config.fig_path, "frac_entropy_explained_by_syn_nearest_neighbour_%s.png" % seed)
        plots.plot_frac_entropy_explained_by(pd.DataFrame(assembly_mi), "Synapse nearest neighbour from assembly", fig_name)

    return assembly_nnds_dict


def assembly_prob_from_indegree_and_syn_nnd(config, assembly_indegrees_dict, assembly_nnds_dict,
                                            colors, min_samples=100):
    """Combines previous results and weights indegrees with synapse neighbour distances
    (and then predicts assembly membership from that for all assemblies)"""

    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)
    labels = list(colors.keys())

    for seed, assembly_grp in assembly_grp_dict.items():
        assembly_indegrees, assembly_nnds = assembly_indegrees_dict[seed], assembly_nnds_dict[seed]
        chance_levels = {}
        assembly_probs = {assembly.idx[0]: {} for assembly in assembly_grp.assemblies}
        bin_centers_dict = {assembly.idx[0]: {} for assembly in assembly_grp.assemblies}
        for assembly in assembly_grp.assemblies:
            assembly_id = assembly.idx[0]
            binned_assembly_nnds = pd.qcut(assembly_nnds.loc[assembly_nnds[assembly_id] > 0, assembly_id],
                                           len(labels), labels=labels)
            idx = np.in1d(binned_assembly_nnds.index.to_numpy(), assembly.gids, assume_unique=True)
            chance_levels[assembly_id] = idx.sum() / len(idx)
            for label_ in labels:
                gids = binned_assembly_nnds.loc[binned_assembly_nnds == label_].index.to_numpy()
                label_assembly_indegrees = assembly_indegrees.loc[gids, assembly_id]
                bin_edges, bin_centers = utils.determine_bins(*np.unique(label_assembly_indegrees.to_numpy(),
                                                                         return_counts=True), min_samples)
                bin_centers_dict[assembly_id][label_] = bin_centers
                bin_idx = np.digitize(label_assembly_indegrees.to_numpy(), bin_edges, right=True)
                gids_tmp, probs = label_assembly_indegrees.index.to_numpy(), []
                for i, center in enumerate(bin_centers):
                    idx = np.in1d(gids_tmp[bin_idx == i + 1], assembly.gids, assume_unique=True)
                    probs.append(idx.sum() / len(idx))
                assembly_probs[assembly_id][label_] = np.array(probs)

        fig_name = os.path.join(config.fig_path, "assembly_prob_from_indegree_syn_nnd_%s.png" % seed)
        plots.plot_assembly_prob_from(bin_centers_dict, assembly_probs, chance_levels,
                                      "Innervation by assembly (weighted by synapse nnd.)", fig_name, colors)


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
    pattern_indegrees, all_gids = get_pattern_innervation(config)
    binned_gids, bin_centers = _bin_gids_by_innervation(pattern_indegrees, all_gids, min_samples)

    for seed, assembly_grp in assembly_grp_dict.items():
        assembly_probs = {pattern_name: {} for pattern_name in list(pattern_indegrees.keys())}
        assembly_mi = {pattern_name: {} for pattern_name in list(pattern_indegrees.keys())}
        for assembly in assembly_grp.assemblies:
            for pattern_name, binned_gids_tmp in binned_gids.items():
                probs, counts, vals = [], [], []
                for bin_center in bin_centers[pattern_name]:
                    idx = np.in1d(binned_gids_tmp[bin_center], assembly.gids, assume_unique=True)
                    probs.append(idx.sum() / len(idx))
                    counts.append(len(binned_gids_tmp[bin_center]))
                    vals.append(bin_center)
                assembly_probs[pattern_name][assembly.idx[0]] = np.array(probs)
                me, pe = _mi_implementation(counts, probs)
                assembly_mi[pattern_name][assembly.idx[0]] = (1.0 - pe / me) * _sign_of_correlation(vals, probs)

        fig_name = os.path.join(config.fig_path, "assembly_prob_from_patterns_%s.png" % seed)
        plots.plot_assembly_prob_from_patterns(bin_centers, assembly_probs, fig_name)
        fig_name = os.path.join(config.fig_path, "frac_entropy_explained_by_patterns_%s.png" % seed)
        plots.plot_frac_entropy_explained_by(pd.DataFrame(assembly_mi), "Innervation by pattern", fig_name)


if __name__ == "__main__":
    config = Config("../configs/v7_bbp-workflow.yaml")
    # assembly_efficacy(config)
    # assembly_in_degree(config)
    # assembly_simplex_counts(config)
    assembly_indegrees = frac_entropy_explained_by_indegree(config)
    assembly_nnds = frac_entropy_explained_by_syn_nnd(config)
    assembly_prob_from_indegree_and_syn_nnd(config, assembly_indegrees, assembly_nnds,
                                            {"below avg.": "assembly_color", "avg.": "gray", "above avg.": "black"})
    # frac_entropy_explained_by_patterns(config)

