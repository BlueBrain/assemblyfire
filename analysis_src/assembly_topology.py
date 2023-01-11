"""
Connectivity analysis (mostly wrappers around `assemblyfire.topology` functions)
last modified: András Ecker 01.2023
"""

import os
from tqdm import tqdm
import numpy as np
import pandas as pd

from assemblyfire.config import Config
import assemblyfire.utils as utils
import assemblyfire.topology as topology
import assemblyfire.plots as plots


def assembly_efficacy(config, assembly_grp_dict):
    """Plots synapses initialized at depressed (rho=0) and potentiated (rho=1) states"""
    c = utils.get_bluepy_circuit(utils.get_sim_path(config.root_path).iloc[0])
    rhos = utils.get_rho0s(c, config.target)  # get all rhos in one go and then index them as needed

    for seed, assembly_grp in tqdm(assembly_grp_dict.items(), desc="Getting efficacies"):
        efficacies = {assembly.idx[0]: rhos.loc[rhos["pre_gid"].isin(assembly.gids)
                                                & rhos["post_gid"].isin(assembly.gids), "rho"].value_counts()
                      for assembly in assembly_grp.assemblies}
        fig_name = os.path.join(config.fig_path, "efficacy_%s.png" % seed)
        plots.plot_efficacy(efficacies, fig_name)


def assembly_in_degrees(assembly_grp_dict, conn_mat, fig_path):
    """Plots indegrees within the assemblies and in their control models"""
    in_degrees, in_degrees_control = topology.in_degree_assemblies(assembly_grp_dict, conn_mat)
    for seed, in_degree in in_degrees.items():
        fig_name = os.path.join(fig_path, "in_degrees_%s.png" % seed)
        plots.plot_in_degrees(in_degree, in_degrees_control[seed], fig_name)


def assembly_simplex_counts(assembly_grp_dict, conn_mat, fig_path):
    """Loads in assemblies and plots simplex counts in assemblies and control models"""
    simplex_counts, simplex_counts_control = topology.simplex_counts_assemblies(assembly_grp_dict, conn_mat)
    for seed, simplices in simplex_counts.items():
        fig_name = os.path.join(fig_path, "simplex_counts_%s.png" % seed)
        plots.plot_simplex_counts(simplices, simplex_counts_control[seed], fig_name)


def assembly_prob_mi_from_indegree(assembly_grp_dict, conn_mat, fig_path, n_bins=21, bin_min_n=10, sign_th=2):
    """Plots assembly probabilities and (relative) fraction of entropy explained from indegrees"""
    gids = conn_mat.gids
    # assembly_indegrees_dict = {}
    for seed, assembly_grp in assembly_grp_dict.items():
        assembly_indegrees = {assembly.idx[0]: conn_mat.degree(assembly.gids, gids)
                              for assembly in assembly_grp.assemblies}
        # assembly_indegrees_dict[seed] = pd.DataFrame(assembly_indegrees, index=gids)
        binned_gids, bin_centers, bin_idx = topology.bin_gids_by_innervation(assembly_indegrees, gids, n_bins)

        plot_args = topology.assembly_membership_probability(gids, assembly_grp, binned_gids, bin_centers, bin_min_n)
        fig_name = os.path.join(fig_path, "assembly_prob_from_indegree_%s.png" % seed)
        palette = {assembly.idx[0]: "pre_assembly_color" for assembly in assembly_grp.assemblies}
        plots.plot_assembly_prob_from(*plot_args, "In degree", palette, fig_name)

        mi = topology.assembly_rel_frac_entropy_explained(gids, assembly_grp, bin_centers, bin_idx,
                                                          seed, bin_min_n, sign_th)
        fig_name = os.path.join(fig_path, "frac_entropy_explained_by_recurrent_innervation_%s.png" % seed)
        plots.plot_frac_entropy_explained_by(mi, "Innervation by assembly", fig_name)

    # return assembly_indegrees_dict


def assembly_prob_mi_from_sinks(assembly_grp_dict, conn_mat, palette, fig_path, n_bins=21, bin_min_n=10, sign_th=2):
    """Plots assembly probabilities and (relative) fraction of entropy explained from generalized in degrees
    (sinks of high dim. simplices). Simplices are found in a way that all non-sink neurons
    are guaranteed to be within the assembly"""
    gids = conn_mat.gids
    dims = list(palette.keys())
    for seed, assembly_grp in assembly_grp_dict.items():
        # building simplex sink count lookup
        assembly_simplices = {dim: {} for dim in dims}
        for assembly in tqdm(assembly_grp.assemblies, desc="Getting assembly simplex lists:"):
            simplex_list = conn_mat.simplex_list(assembly.gids, gids)
            for dim in dims:
                sink_counts, _ = np.histogram(simplex_list[dim][:, -1], np.arange(len(gids) + 1))
                assembly_simplices[dim][assembly.idx[0]] = sink_counts

        plot_args = [{dim: {} for dim in dims}, {dim: {} for dim in dims}, {dim: {} for dim in dims}, {dim: {} for dim in dims}]
        for dim in dims:
            binned_gids, bin_centers, bin_idx = topology.bin_gids_by_innervation(assembly_simplices[dim], gids, n_bins)
            # getting MI matrices for all dimensions:
            mi = topology.assembly_rel_frac_entropy_explained(gids, assembly_grp, bin_centers, bin_idx,
                                                              seed, bin_min_n, sign_th)
            fig_name = os.path.join(fig_path, "frac_entropy_explained_by_%iDsimplex_sinks_%s.png" % (dim, seed))
            plots.plot_frac_entropy_explained_by(mi, "(Generalized) Innervation by assembly", fig_name)
            # gathering data for a simplified plot (not the best looking code... but it does the job)
            plot_args_dim = topology.assembly_membership_probability(gids, assembly_grp, binned_gids, bin_centers, bin_min_n)
            for i in range(len(plot_args_dim)-1):
                for assembly_id in list(plot_args_dim[0].keys()):
                    plot_args[i][dim][assembly_id] = plot_args_dim[i][assembly_id][assembly_id]
            if dim == dims[0]:
                plot_args.append(plot_args_dim[-1])
        # one plot across dimensions (but only for within assembly simplices)
        fig_name = os.path.join(fig_path, "assembly_prob_from_simplex_dim_%s.png" % seed)
        plots.plot_assembly_prob_from(*plot_args, "Generalized in degree (#simplex sinks)", palette, fig_name, True)


'''TODO: replace this with Michael's implementation
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
'''


def _get_spiking_proj_gids(config, sim_config):
    """Loads grouped (to patterns + non-specific) TC gids
    (Could be done easier with adding stuff to the yaml config... but whatever)"""
    tc_spikes = utils.get_grouped_tc_spikes(config.pattern_gids_fname, sim_config, config.t_start, config.t_end)
    _, patterns = utils.get_stimulus_stream(config.input_patterns_fname, config.t_start, config.t_end)
    pattern_names = np.unique(patterns)
    projf_names = list(utils.get_projf_names(sim_config).keys())
    assert len(projf_names) <= 2, "The code assumes max 2 projections, one pattern specific and one non-specific"
    ns_projf_name = np.setdiff1d(projf_names, [config.patterns_projection_name])[0]
    pattern_gids, ns_gids = {}, []
    for name, data in tc_spikes.items():
        if name in pattern_names:
            pattern_gids[name] = np.unique(data["spiking_gids"])
        else:
            ns_gids.extend(np.unique(data["spiking_gids"]))
    all_pattern_gids = []
    for pattern_name, gids in pattern_gids.items():
        all_pattern_gids.extend(gids)
    return {config.patterns_projection_name: np.unique(all_pattern_gids), ns_projf_name: np.unique(ns_gids)}, pattern_gids


def get_proj_innervation(config):
    """Looks up how many projection fibers, and pattern fibers innervate the neurons"""
    from conntility.circuit_models import circuit_connection_matrix

    sim = utils.get_bluepy_simulation(utils.get_sim_path(config.root_path).iloc[0])
    proj_gids, pattern_gids = _get_spiking_proj_gids(config, sim.config)
    c = sim.circuit
    post_gids = utils.get_gids(c, config.target)
    proj_indegrees, pattern_indegrees, mutual_innervation_matrices = {}, {}, {}
    for proj, pre_gids in proj_gids.items():
        # get (sparse) connectivity matrix between the input fibers and neurons in the circuit
        input_conn_mat = circuit_connection_matrix(c, proj, pre_gids, post_gids).tocsr()
        mutual_innervation_matrices[proj] = input_conn_mat.transpose() * input_conn_mat
        proj_indegrees[proj] = np.array(input_conn_mat.sum(axis=0)).flatten()
        if proj == config.patterns_projection_name:
            # for each pattern get how many pattern fibers innervate the neurons
            for pattern_name, gids in pattern_gids.items():
                pattern_idx = np.in1d(pre_gids, gids, assume_unique=True)
                pattern_indegrees[pattern_name] = np.array(input_conn_mat[pattern_idx].sum(axis=0)).flatten()

    return proj_indegrees, pattern_indegrees, mutual_innervation_matrices, post_gids


def n_assembly_prob_from_projs(assembly_grp_dict, proj_indegrees, gids, fig_path, n_bins=21, bin_min_n=10):
    """Plots number of assemblies a neuron participates in vs. indegree from projections"""
    for seed, assembly_grp in assembly_grp_dict.items():
        n_assemblies = np.sum(assembly_grp.as_bool(), axis=1)
        # as assembly groups only store info about spiking gids,
        # we'll need to get rid of the non-spiking ones from the degree vectors...
        spiking_gid_idx = np.in1d(gids, assembly_grp.all, assume_unique=True)
        proj_indegrees_seed = {proj_name: indegrees[spiking_gid_idx] for proj_name, indegrees in proj_indegrees.items()}
        bin_centers_dict, assembly_probs, assembly_probs_low, assembly_probs_high = {}, {}, {}, {}
        for proj_name, indegrees in proj_indegrees_seed.items():
            # same edges as used in `topology.bin_gids_by_innervation()`
            bin_edges = np.hstack(([0], np.linspace(np.percentile(indegrees[indegrees != 0], 1),
                                                    np.percentile(indegrees[indegrees != 0], 99), n_bins)))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_centers_dict[proj_name] = bin_centers
            bin_idx = np.digitize(indegrees, bin_edges, right=True)
            probs = np.zeros_like(bin_centers, dtype=np.float32)
            probs_low, probs_high = np.zeros_like(probs), np.zeros_like(probs)
            for i in range(len(bin_centers)):
                probs[i], probs_low[i], probs_high[i] = topology.prob_with_binom_ci(n_assemblies[bin_idx == i + 1], bin_min_n)
            assembly_probs[proj_name] = probs
            assembly_probs_low[proj_name], assembly_probs_high[proj_name] = probs_low, probs_high
        fig_name = os.path.join(fig_path, "assembly_n_from_projections_seed%i.png" % seed)
        plot_assembly_n_from(bin_centers_dict, assembly_probs, assembly_probs_low, assembly_probs_high,
                             "In degree from projections", "projections", fig_name)


def assembly_prob_mi_from_patterns(assembly_grp_dict, pattern_indegrees, gids, fig_path,
                                   n_bins=21, bin_min_n=10, sign_th=2):
    """Plots assembly probabilities and (relative) fraction of entropy explained from pattern indegrees"""
    binned_gids, bin_centers, bin_idx = topology.bin_gids_by_innervation(pattern_indegrees, gids, n_bins)
    for seed, assembly_grp in assembly_grp_dict.items():
        plot_args = topology.assembly_membership_probability(gids, assembly_grp, binned_gids, bin_centers, bin_min_n)
        fig_name = os.path.join(fig_path, "assembly_prob_from_patterns_%s.png" % seed)
        plots.plot_assembly_prob_from(*plot_args, "In degree from patterns", "patterns", fig_name)

        mi = topology.assembly_rel_frac_entropy_explained(gids, assembly_grp, bin_centers, bin_idx,
                                                          seed, bin_min_n, sign_th)
        fig_name = os.path.join(fig_path, "frac_entropy_explained_by_patterns_%s.png" % seed)
        plots.plot_frac_entropy_explained_by(mi, "Innervation by pattern", fig_name)


if __name__ == "__main__":
    config = Config("../configs/v7_10seeds_np.yaml")
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)
    conn_mat = topology.AssemblyTopology.from_h5(config.h5f_name, prefix=config.h5_prefix_connectivity)

    # assembly_efficacy(config, assembly_grp_dict)
    assembly_in_degrees(assembly_grp_dict, conn_mat, config.fig_path)
    assembly_simplex_counts(assembly_grp_dict, conn_mat, config.fig_path)
    assembly_prob_mi_from_indegree(assembly_grp_dict, conn_mat, config.fig_path)
    assembly_prob_mi_from_sinks(assembly_grp_dict, conn_mat, {2: "gray", 3: "black", 4: "assembly_color"},
                                config.fig_path)
    # assembly_nnds = frac_entropy_explained_by_syn_nnd(config)
    # assembly_prob_from_indegree_and_syn_nnd(config, assembly_indegrees, assembly_nnds,
    #                                         {"below avg.": "assembly_color", "avg.": "gray", "above avg.": "black"})

    proj_indegrees, pattern_indegrees, mutual_innervation_matrices, gids = get_proj_innervation(config)
    n_assembly_prob_from_projs(assembly_grp_dict, proj_indegrees, gids, config.fig_path)
    assembly_prob_mi_from_patterns(assembly_grp_dict, pattern_indegrees, gids, config.fig_path)

