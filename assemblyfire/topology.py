"""
Advanced network metrics on ConnectivityMatrix (now in `conntility`)
authors: Daniela Egas Santander, Nicolas Ninin, András Ecker
last modified: 01.2023
"""

from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import binom
from pyitlib import discrete_random_variable as drv

from conntility.connectivity import ConnectivityMatrix


class AssemblyTopology(ConnectivityMatrix):
    """
    A class derived from ConnectivityMatrix with additional information on networks metrics
    of the subgraph associated to an assembly within the connectivity matrix of the circuit.
    """

    def degree(self, pre_gids=None, post_gids=None, kind="in"):
        """Returns in/out degrees of the (symmetric) subarray specified by `pre_gids`
        (if `post_gids` is given as well, then the subarray will be asymmetric)"""
        if pre_gids is None:
            matrix = self.matrix
        else:
            if post_gids is None:
                matrix = self.submatrix(pre_gids)
            else:
                matrix = self.submatrix(pre_gids, sub_gids_post=post_gids)
        if kind == "in":
            return np.array(matrix.sum(axis=0)).flatten()
        elif kind == "out":
            return np.array(matrix.sum(axis=1)).flatten()
        else:
            ValueError("Need to specify 'in' or 'out' degree!")

    def density(self, sub_gids=None):
        """Returns the density of submatrix specified by `sub_gids`"""
        matrix = self.matrix if sub_gids is None else self.submatrix(sub_gids)
        return matrix.getnnz()/np.prod(matrix.shape)

    def simplex_counts(self, sub_gids=None):
        """Returns the simplex counts of submatrix specified by `sub_gids`"""
        from pyflagser import flagser_count_unweighted
        matrix = self.matrix if sub_gids is None else self.submatrix(sub_gids)
        return flagser_count_unweighted(matrix, directed=True)

    def simplex_list(self, pre_gids=None, post_gids=None):
        """Returns the simplex list of submatrix specified by `sub_gids`"""
        from scipy.sparse import coo_matrix
        from pyflagsercount import flagser_count
        if pre_gids is None:
            matrix = self.matrix
        else:
            if post_gids is None:
                matrix = self.submatrix(pre_gids)
            else:
                row_gids = self.gids[self._edge_indices["row"].to_numpy()]
                if (post_gids == self.gids).all():
                    keep_idx = np.in1d(row_gids, pre_gids)
                else:
                    col_gids = self.gids[self._edge_indices["col"].to_numpy()]
                    keep_idx = np.in1d(row_gids, pre_gids) & np.in1d(col_gids, post_gids)
                matrix = coo_matrix((self.edges["data"][keep_idx].to_numpy(),
                                     (self._edge_indices["row"][keep_idx].to_numpy(),
                                      self._edge_indices["col"][keep_idx].to_numpy())),
                                    shape=(len(self.gids), len(self.gids)))
        flagser = flagser_count(matrix, return_simplices=True, max_simplices=False)
        return [np.array(x) for x in flagser["simplices"]]

    def betti_counts(self, sub_gids=None):
        """Returns the betti counts of submatrix specified by `sub_gids`"""
        from pyflagser import flagser_unweighted
        matrix = self.matrix if sub_gids is None else self.submatrix(sub_gids)
        return flagser_unweighted(matrix, directed=True)["betti"]


def in_degree_assemblies(assembly_grp_dict, conn_mat, post_id=None):
    """
    Computes the in degree distribution within assemblies (or cross-assemblies if `post_assembly_id` is specified)
    across seeds and a random controls of the same size/depth profile/mtype composition
    :param assembly_grp_dict: dict with seeds as keys and assembly groups as values
    :param conn_mat: AssemblyTopology object for the circuit where the assemblies belong to
    :param post_id: optional ID of postsynaptic assembly for cross-assembly in degrees
    :return in_degrees: dict with the same keys as `assembly_grp_dict` - within that another dict
                        with keys as assembly labels and list of in degrees as values
    :return in_d_control: dict with the same keys as `assembly_grp_dict` - within that another dict
                          with keys ['n', 'depths', 'mtype'] and yet another dict similar to `in_degrees`
                          but values are in degrees of the random controls
    """
    in_degrees = {}
    in_d_control = {seed: {} for seed in list(assembly_grp_dict.keys())}
    for seed, assembly_grp in tqdm(assembly_grp_dict.items(), desc="Getting in-degrees"):
        post_gids = assembly_grp.loc((post_id, int(seed.split("seed")[1]))).gids if post_id is not None else None
        in_degrees[seed] = {assembly.idx: conn_mat.degree(assembly.gids, post_gids, kind="in")
                            for assembly in assembly_grp.assemblies}
        in_d_control[seed]["n"] = {assembly.idx: conn_mat.degree(conn_mat.random_n_gids(assembly.gids), post_gids,
                                   kind="in") for assembly in assembly_grp.assemblies}
        in_d_control[seed]["depths"] = {assembly.idx: conn_mat.degree(
                                        conn_mat.index("depth").random_numerical_gids(assembly.gids), post_gids,
                                        kind="in") for assembly in assembly_grp.assemblies}
        in_d_control[seed]["mtypes"] = {assembly.idx: conn_mat.degree(
                                        conn_mat.index("mtype").random_categorical_gids(assembly.gids), post_gids,
                                        kind="in") for assembly in assembly_grp.assemblies}
    return in_degrees, in_d_control


def simplex_counts_assemblies(assembly_grp_dict, conn_mat):
    """
    Computes the number of simplices of assemblies across seeds
    and a random controls of the same size/depth profile/mtype composition
    :param assembly_grp_dict: dict with seeds as keys and assembly groups as values
    :param conn_mat: AssemblyTopology object for the circuit where the assemblies belong to
    :return simplex_count: dict with the same keys as `assembly_grp_dict` - within that another dict
                           with keys as assembly labels and list of simplex counts as values
    :return simplex_counts_control: dict with the same keys as `assembly_grp_dict` - within that another dict
                                    with keys ['n', 'depths', 'mtype'] and yet another dict similar to `simplex_count`
                                    but values are simplex counts of the random controls
    """
    simplex_counts = {}
    s_c_control = {seed: {} for seed in list(assembly_grp_dict.keys())}
    for seed, assembly_grp in tqdm(assembly_grp_dict.items(), desc="Counting simplices"):
        simplex_counts[seed] = {assembly.idx: conn_mat.simplex_counts(assembly.gids)
                                for assembly in assembly_grp.assemblies}
        s_c_control[seed]["n"] = {assembly.idx: conn_mat.simplex_counts(conn_mat.random_n_gids(assembly.gids))
                                  for assembly in assembly_grp.assemblies}
        s_c_control[seed]["depths"] = {assembly.idx: conn_mat.simplex_counts(
                                       conn_mat.index("depth").random_numerical_gids(assembly.gids))
                                       for assembly in assembly_grp.assemblies}
        s_c_control[seed]["mtypes"] = {assembly.idx: conn_mat.simplex_counts(
                                       conn_mat.index("mtype").random_categorical_gids(assembly.gids))
                                       for assembly in assembly_grp.assemblies}
    return simplex_counts, s_c_control


def simplex_counts_consensus_instantiations(consensus_assemblies_dict, conn_mat):
    """Computes the simplices of all assemblies making up the consensus assemblies
    and a random control of the size of the average of instantiations"""
    simplex_count, s_c_control = {}, {}
    for k, consensus_assembly in tqdm(consensus_assemblies_dict.items(), desc="Counting simplices"):
        # Simplex count of instantiations within one cluster (under one key)
        simplex_count[k] = [conn_mat.simplex_counts(inst.gids) for inst in consensus_assembly.instantiations]
        # Comparison with random control of average size of the instantiations
        mean_size = int(np.mean([len(inst.gids) for inst in consensus_assembly.instantiations]))
        sample_gids = np.random.choice(conn_mat.gids, mean_size, replace=False)
        s_c_control[k] = [conn_mat.simplex_counts(sample_gids)]
    return simplex_count, s_c_control


# TODO: maybe this should be moved to the ConsensusAssembly class
def get_intersection_gids(consensus_assemblies_dict):
    """Returns dictionary of intersection gids"""
    intersection_gids_dict = {}
    for k, consensus_assembly in consensus_assemblies_dict.items():
        max_filtration = np.max(consensus_assembly.coreness)
        intersection_gids_dict[k] = consensus_assembly.union.gids[
                                    np.where(consensus_assembly.coreness == max_filtration)]
        # TODO: Implement the above in a weighted assembly class where you can choose thresholds
    return intersection_gids_dict


def simplex_counts_consensus_core_union_intersection(consensus_assemblies_dict, conn_mat):
    """Computes the simplex counts of the core and intersection of the consensus assemblies"""
    intersection_gids_dict = get_intersection_gids(consensus_assemblies_dict)
    s_c_core, s_c_union, s_c_intersection = {}, {}, {}
    for k, consensus_assembly in tqdm(consensus_assemblies_dict.items(), desc="Counting simplices"):
        s_c_core[k] = conn_mat.simplex_counts(consensus_assembly)
        s_c_union[k] = conn_mat.simplex_counts(consensus_assembly.union)
        s_c_intersection[k] = conn_mat.simplex_counts(intersection_gids_dict[k])
    return s_c_core, s_c_intersection


def bin_gids_by_innervation(all_indegrees, gids, n_bins):
    """Creates lookups of gids in optimal bins for each pre-synaptic group (in terms of in-degree)
    works with both dictionary and DataFrame (column-wise)"""
    binned_gids, bin_centers_dict, bin_idx_dict = {key: {} for key in list(all_indegrees.keys())}, {}, {}
    for key, indegrees in all_indegrees.items():
        if isinstance(indegrees, pd.Series):
            indegrees = indegrees.to_numpy()
        idx = np.where(indegrees >= 0.)[0]  # sometimes -1s are used as placeholders...
        gids_tmp, indegrees = gids[idx], indegrees[idx]
        bin_edges = np.hstack(([0], np.linspace(np.percentile(indegrees[indegrees != 0], 1),
                                                np.percentile(indegrees[indegrees != 0], 99), n_bins)))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_centers_dict[key] = bin_centers
        bin_idx = np.digitize(indegrees, bin_edges, right=True)
        bin_idx_dict[key] = bin_idx
        for i, center in enumerate(bin_centers):
            binned_gids[key][center] = gids_tmp[bin_idx == i+1]
    return binned_gids, bin_centers_dict, bin_idx_dict


def _prob_with_binom_ci(samples, min_n):
    """Probability (just as the mean of samples) and binomial distribution based confidence interval"""
    samples = samples.astype(bool)
    n_samples = len(samples)
    if n_samples < min_n:
        return np.nan, np.nan, np.nan
    p = np.linspace(0, 1, 100)
    p_n_cond_p = binom(n_samples, p).pmf(np.sum(samples))
    p_p_post = p_n_cond_p / np.sum(p_n_cond_p)
    return np.mean(samples), np.interp(0.05, np.cumsum(p_p_post), p), np.interp(0.95, np.cumsum(p_p_post), p)


def assembly_membership_probability(gids, assembly_grp, binned_gids, bin_centers, bin_min_n):
    """Gets membership probability (and CI) for all gids in all assemblies (using pre-binned indegrees)"""
    chance_levels = {}
    keys = list(binned_gids.keys())
    bin_centers_plot, assembly_probs = {key: {} for key in keys}, {key: {} for key in keys}
    assembly_probs_low, assembly_probs_high = {key: {} for key in keys}, {key: {} for key in keys}
    for assembly in assembly_grp.assemblies:
        assembly_id = assembly.idx[0]
        idx = np.in1d(gids, assembly.gids, assume_unique=True)
        chance_levels[assembly_id] = np.sum(idx) / len(idx)
        for key, binned_gids_tmp in binned_gids.items():
            bin_centers_plot[key][assembly_id] = bin_centers[key]
            probs = np.zeros_like(bin_centers[key], dtype=np.float32)
            probs_low, probs_high = np.zeros_like(probs), np.zeros_like(probs)
            for i, bin_center in enumerate(bin_centers[key]):
                idx = np.in1d(binned_gids_tmp[bin_center], assembly.gids, assume_unique=True)
                probs[i], probs_low[i], probs_high[i] = _prob_with_binom_ci(idx, bin_min_n)
            assembly_probs[key][assembly_id] = probs
            assembly_probs_low[key][assembly_id] = probs_low
            assembly_probs_high[key][assembly_id] = probs_high
    return bin_centers_plot, assembly_probs, assembly_probs_low, assembly_probs_high, chance_levels


def _sign(bin_centers, probs, counts=None):
    """Gets sign of line fitted to assembly probs. vs. indegree"""
    return np.sign(np.polyfit(bin_centers, probs, 1, w=counts)[0])


def assembly_rel_frac_entropy_explained(gids, assembly_grp, bin_centers, bin_idx, seed, bin_min_n, sign_th):
    """Gets mutual information between assembly membership and structural innervation (using pre-binned indegrees)"""
    if isinstance(seed, str):
        seed = int(seed.split("seed")[1])
    keys = np.sort(list(bin_idx.keys()))
    assembly_idx = np.sort([assembly.idx[0] for assembly in assembly_grp.assemblies])
    mi_matrix = np.zeros((len(keys), len(assembly_idx)), dtype=np.float32)
    mi_ctrl_matrix = np.zeros_like(mi_matrix)
    for j, assembly_id in enumerate(assembly_idx):
        for i, key in enumerate(keys):
            idx = np.in1d(gids, assembly_grp.loc((assembly_id, seed)).gids, assume_unique=True)
            bin_idx_ = bin_idx[key].copy()
            mi = drv.information_mutual(idx, bin_idx_) / drv.entropy(idx)
            mi_ctrl = drv.information_mutual(idx, np.random.permutation(bin_idx_)) / drv.entropy(idx)
            # recalculate assembly probability for line fitting
            # (could be passed from `get_assembly_membership_probability()` but whatever...)
            probs = np.zeros_like(bin_centers[key], dtype=np.float32)
            counts = np.zeros_like(probs, dtype=int)
            for k, center in enumerate(bin_centers[key]):
                tmp = idx[bin_idx_ == k + 1]
                counts[k], probs[k] = len(tmp), np.mean(tmp)
            valid_n_idx = np.where(counts >= bin_min_n)
            mi_sign = _sign(bin_centers[key][valid_n_idx], probs[valid_n_idx], counts[valid_n_idx])
            mi_matrix[i, j] = mi_sign * mi
            mi_ctrl_matrix[i, j] = mi_ctrl
    # set values that are smaller than control mean + significance threshold * control std to nan...
    if sign_th > 0:
        mi_matrix[np.abs(mi_matrix) < (np.mean(mi_ctrl_matrix) + sign_th * np.std(mi_ctrl_matrix))] = np.nan
    ratio = (np.nanmean(np.abs(mi_matrix)) - np.mean(mi_ctrl_matrix)) / np.std(mi_ctrl_matrix)
    print("MI ratio (between data and shuffled/control data): %.2f" % ratio)
    return pd.DataFrame(data=mi_matrix, columns=assembly_idx, index=keys)

