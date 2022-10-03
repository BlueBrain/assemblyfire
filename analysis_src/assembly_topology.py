"""
In degrees, simplex counts, and pot/dep ratios of assemblies
last modified: András Ecker 09.2022
"""

import os
from tqdm import tqdm
import numpy as np

import assemblyfire.utils as utils
from assemblyfire.config import Config
from assemblyfire.topology import AssemblyTopology, in_degree_assemblies, simplex_counts_assemblies
from assemblyfire.plots import plot_efficacy, plot_in_degrees, plot_assembly_prob_from_indegree,\
                               plot_simplex_counts, plot_assembly_prob_from_innervation, plot_frac_entropy_explained_by_innervation


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
        plot_efficacy(efficacies, fig_name)


def assembly_in_degree(config):
    """Loads in assemblies and plots in degrees within assemblies and cross assemblies seed-by-seed
    (for cross assembly in degrees the postsynaptic target assembly is fixed: the late assembly)"""

    conn_mat = AssemblyTopology.from_h5(config.h5f_name,
                                        prefix=config.h5_prefix_connectivity, group_name="full_matrix")
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)

    in_degrees, in_degrees_control = in_degree_assemblies(assembly_grp_dict, conn_mat)
    for seed, in_degree in in_degrees.items():
        fig_name = os.path.join(config.fig_path, "in_degrees_%s.png" % seed)
        plot_in_degrees(in_degree, in_degrees_control[seed], fig_name)

    '''
    in_degrees, in_degrees_control = in_degree_assemblies(assembly_grp_dict, conn_mat, post_id=0)
    for seed, in_degree in in_degrees.items():
        fig_name = os.path.join(config.fig_path, "cross_assembly_in_degrees_%s.png" % seed)
        plot_in_degrees(in_degree, in_degrees_control[seed], fig_name, xlabel="Cross assembly (any to 0) in degree")
    '''


def assembly_prob_from_indegree(config, min_samples=100):
    """Loads in assemblies and for each of them plots the probabilities of assembly membership
    vs. in degree (from the assembly neurons)"""

    conn_mat = AssemblyTopology.from_h5(config.h5f_name,
                                        prefix=config.h5_prefix_connectivity, group_name="full_matrix")
    all_gids = conn_mat.vertices["gid"].to_numpy()
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)

    for seed, assembly_grp in tqdm(assembly_grp_dict.items(), desc="Getting prob. vs. in-degrees"):
        bin_centers_dict, assembly_probs = {}, {}
        for assembly in assembly_grp.assemblies:
            in_degrees = conn_mat.degree(assembly.gids, all_gids)
            bin_edges, bin_centers = utils.determine_bins(*np.unique(in_degrees, return_counts=True), min_samples)
            bin_idx = np.digitize(in_degrees, bin_edges, right=True)
            probs = []
            for i, center in enumerate(bin_centers):
                idx = np.in1d(all_gids[bin_idx == i + 1], assembly.gids, assume_unique=True)
                probs.append(idx.sum() / len(idx))
            bin_centers_dict[assembly.idx[0]] = bin_centers
            assembly_probs[assembly.idx[0]] = np.array(probs)

        fig_name = os.path.join(config.fig_path, "assembly_prob_from_indegree_%s.png" % seed)
        plot_assembly_prob_from_indegree(bin_centers_dict, assembly_probs, fig_name)


def assembly_simplex_counts(config):
    """Loads in assemblies and plots simplex counts (seed by seed
    and then for all instantiations per consensus assemblies)"""

    conn_mat = AssemblyTopology.from_h5(config.h5f_name,
                                        prefix=config.h5_prefix_connectivity, group_name="full_matrix")
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)

    simplex_counts, simplex_counts_control = simplex_counts_assemblies(assembly_grp_dict, conn_mat)
    for seed, simplices in simplex_counts.items():
        fig_name = os.path.join(config.fig_path, "simplex_counts_%s.png" % seed)
        plot_simplex_counts(simplices, simplex_counts_control[seed], fig_name)


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
    pattern_nconns = {}
    for pattern_name, gids in pattern_gids.items():
        pattern_idx = np.in1d(pre_gids, gids, assume_unique=True)
        pattern_nconns[pattern_name] = np.array(input_conn_mat[pattern_idx].sum(axis=0)).flatten()
    return pattern_nconns, post_gids


def _bin_gids_by_innervation(pattern_nconns, gids, min_samples):
    """Creates lookups of gids in optimal bins for each pattern
    (optimal bins are determined based on their thalamic innervation profile)"""
    binned_gids, bin_centers_dict = {pattern_name: {} for pattern_name in list(pattern_nconns.keys())}, {}
    for pattern_name, nconns in pattern_nconns.items():
        bin_edges, bin_centers = utils.determine_bins(*np.unique(nconns, return_counts=True), min_samples)
        bin_centers_dict[pattern_name] = bin_centers
        bin_idx = np.digitize(nconns, bin_edges, right=True)
        for i, center in enumerate(bin_centers):
            binned_gids[pattern_name][center] = gids[bin_idx == i+1]
    return binned_gids, bin_centers_dict


def assembly_prob_from_innervation(config, min_samples=100):
    """Loads in assemblies and for each of them plots the probabilities of assembly membership
    vs. purely structural innervation by the input patterns"""

    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)
    pattern_nconns, all_gids = get_pattern_innervation(config)
    binned_gids, bin_centers = _bin_gids_by_innervation(pattern_nconns, all_gids, min_samples)

    for seed, assembly_grp in assembly_grp_dict.items():
        assembly_probs = {pattern_name: {} for pattern_name in list(pattern_nconns.keys())}
        for assembly in assembly_grp.assemblies:
            for pattern_name, binned_gids_tmp in binned_gids.items():
                probs = []
                for bin_center in bin_centers[pattern_name]:
                    idx = np.in1d(binned_gids_tmp[bin_center], assembly.gids, assume_unique=True)
                    probs.append(idx.sum() / len(idx))
                assembly_probs[pattern_name][assembly.idx[0]] = np.array(probs)

        fig_name = os.path.join(config.fig_path, "assembly_prob_from_innervation_%s.png" % seed)
        plot_assembly_prob_from_innervation(bin_centers, assembly_probs, fig_name)


def _mi_implementation(degree_counts, degree_p):
    """
    Analyzes how much of the uncertainty of assembly membership is explained away when one considers the strengths
    of innervations from a given pre-synaptic target (in terms of in-degree).

    Args:
        degree_counts (list, or numpy.array): The number of neurons in the simulation that have a given degree. One 
        entry per degree-bin.
        degree_p (list, or numpy.array): The probability that neurons with a given degree are members of the assembly
        in question. Must have same length as degree_counts.
        Note: Yes, for this analysis the actual _value_ of a degree-bin (which min/max degree does it represent?) is
        irrelevant.

    Returns:
        membership_entropy (float): The prior entropy of assembly membership.
        posterior_entropy (float): The posterior entropy of assembly membership conditional on the innervation degree.
    """
    import numpy
    def entropy(p):
        return -numpy.log2(p) * p - numpy.log2(1 - p) * (1 - p)
    
    def entropy_vec(p_vec):
        return numpy.nansum(numpy.vstack([
            -numpy.log2(p_vec) * p_vec,
            -numpy.log2(1 - p_vec) * (1 - p_vec)
        ]), axis=0)
    
    degree_counts = numpy.array(degree_counts); degree_p = numpy.array(degree_p)
    
    overall_p = (degree_counts * degree_p).sum() / degree_counts.sum()
    membership_entropy = entropy(overall_p)
    
    posterior_entropy = (entropy_vec(degree_p) * degree_counts).sum() / degree_counts.sum()
    
    return membership_entropy, posterior_entropy


def sign_of_correlation(degree_vals, degree_p):
    """
    Analyzes whether the strength of innervation from a given pre-synaptic target (in terms of in-degree) is rather 
    increasing (positive sign) or decreasing (negative sign) the probability that the innervated neuron is member of
    an assembly.

    Args:
        degree_vals (list, or numpy.array): The possible values of degrees for the innervated neurons. E.g. the centers
        of degree-bins.
        degree_p (list, or numpy.array): The probability that neurons with a given degree are members of the assembly
        in question. Must have same length as degree_vals.
    
    Returns: 
        sign (int): -1 if stronger innervation decreases probability of membership; 1 if it rather increases it
    """
    import numpy
    degree_vals = numpy.array(degree_vals); degree_p = numpy.array(degree_p)
    idxx = numpy.argsort(degree_vals)
    return numpy.sign(numpy.polyfit(degree_vals[idxx], degree_p[idxx], 1)[0])


def fraction_entropy_explained(config, min_samples=100):
    """Contributed by MWR. Re-using a lot of assembly_prob_from_innervation. Better implementation possible"""
    import pandas

    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)
    pattern_nconns, all_gids = get_pattern_innervation(config)
    binned_gids, bin_centers = _bin_gids_by_innervation(pattern_nconns, all_gids, min_samples)

    for seed, assembly_grp in assembly_grp_dict.items():
        assembly_mi = {pattern_name: {} for pattern_name in list(pattern_nconns.keys())}
        
        for assembly in assembly_grp.assemblies:
            for pattern_name, binned_gids_tmp in binned_gids.items():
                probs = []; counts = []; vals = []
                for bin_center in bin_centers[pattern_name]:
                    idx = np.in1d(binned_gids_tmp[bin_center], assembly.gids, assume_unique=True)
                    probs.append(idx.sum() / len(idx))
                    counts.append(len(binned_gids_tmp[bin_center]))
                    vals.append(bin_center)
                me, pe = _mi_implementation(counts, probs)
                assembly_mi[pattern_name][assembly.idx[0]] = (1.0 - pe / me) * sign_of_correlation(vals, probs)
        
        fig_name = os.path.join(config.fig_path, "frac_entropy_explained_by_tc_innervation_%s.png" % seed)
        plot_frac_entropy_explained_by_innervation(pandas.DataFrame(assembly_mi), fig_name)


if __name__ == "__main__":
    config = Config("../configs/v7_bbp-workflow.yaml")
    assembly_efficacy(config)
    assembly_in_degree(config)
    assembly_prob_from_indegree(config)
    assembly_simplex_counts(config)
    assembly_prob_from_innervation(config)
    fraction_entropy_explained(config)
