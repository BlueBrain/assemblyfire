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


def simplex_counts_consensus_instantiations(assembly_grp, conn_mat):
    """Computes the simplices of all assemblies making up the consensus assemblies
    and a random control of the size of the average of instantiations"""
    simplex_count, s_c_control = {}, {}
    for assembly in tqdm(assembly_grp.assemblies, desc="Counting simplices"):
        simplex_count[assembly.idx[0]] = [conn_mat.simplex_counts(inst.gids) for inst in assembly.instantiations]
        ctrl_gids = conn_mat.random_n_gids(int(np.mean([len(inst.gids) for inst in assembly.instantiations])))
        s_c_control[assembly.idx[0]] = [conn_mat.simplex_counts(ctrl_gids)]
    return simplex_count, s_c_control


def bin_gids_by_innervation(all_indegrees, gids, n_bins):
    """Creates lookups of gids in optimal bins for each pre-synaptic group (in terms of in-degree)
    works with both dictionary and DataFrame (column-wise)"""
    binned_gids, bin_centers_dict, bin_idx_dict = {key: {} for key in list(all_indegrees.keys())}, {}, {}
    for key, indegrees in all_indegrees.items():
        if isinstance(indegrees, pd.Series):
            indegrees = indegrees.to_numpy()
        if np.any(indegrees < 0):  # to deal with zsored values...
            idx = ~np.isnan(indegrees)
            bin_edges = np.linspace(np.percentile(indegrees[idx], 1), np.percentile(indegrees[idx], 99), n_bins + 1)
            indegrees[~idx] = np.min(indegrees)  # replace NaNs with minimum value (to not break stuff afterwards)
        else:
            bin_edges = np.hstack(([0], np.linspace(np.percentile(indegrees[indegrees != 0], 1),
                                                    np.percentile(indegrees[indegrees != 0], 99), n_bins)))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_centers_dict[key] = bin_centers
        bin_idx = np.digitize(indegrees, bin_edges, right=True)
        bin_idx_dict[key] = bin_idx
        for i, center in enumerate(bin_centers):
            binned_gids[key][center] = gids[bin_idx == i+1]
    return binned_gids, bin_centers_dict, bin_idx_dict


def prob_with_binom_ci(samples, min_n):
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
        chance_levels[assembly_id] = np.mean(np.in1d(gids, assembly.gids, assume_unique=True))
        for key, binned_gids_tmp in binned_gids.items():
            bin_centers_plot[key][assembly_id] = bin_centers[key]
            probs = np.zeros_like(bin_centers[key], dtype=np.float32)
            probs_low, probs_high = np.zeros_like(probs), np.zeros_like(probs)
            for i, bin_center in enumerate(bin_centers[key]):
                idx = np.in1d(binned_gids_tmp[bin_center], assembly.gids, assume_unique=True)
                probs[i], probs_low[i], probs_high[i] = prob_with_binom_ci(idx, bin_min_n)
            assembly_probs[key][assembly_id] = probs
            assembly_probs_low[key][assembly_id] = probs_low
            assembly_probs_high[key][assembly_id] = probs_high
    return bin_centers_plot, assembly_probs, assembly_probs_low, assembly_probs_high, chance_levels


def cond_assembly_membership_probability(gids, assembly_grp, bin_centers, bin_idx, cond_df, cond_keys, seed, bin_min_n):
    """Reimplementation of `assembly_membership_probability()` above with some changes:
    1: it's not using pre-binned gids, but `bin_idx` (from `np.digitize`) and an extra condition passed in `cond_df`
    2: it doesn't do full cross assembly analysis (because that's a lot with extra condition), only within assembly."""
    if isinstance(seed, str) and seed not in ["consensus", "average"]:
        seed = int(seed.split("seed")[1])
    chance_levels = {}
    keys = list(cond_keys.keys())
    bin_centers_plot, assembly_probs = {key: {} for key in keys}, {key: {} for key in keys}
    assembly_probs_low, assembly_probs_high = {key: {} for key in keys}, {key: {} for key in keys}
    for assembly_id, bin_centers_ in bin_centers.items():
        assembly_gids = assembly_grp.loc((assembly_id, seed)).gids
        chance_levels[assembly_id] = np.mean(np.in1d(gids, assembly_gids, assume_unique=True))
        for key, key_id in cond_keys.items():
            bin_centers_plot[key][assembly_id] = bin_centers_
            probs = np.zeros_like(bin_centers_, dtype=np.float32)
            probs_low, probs_high = np.zeros_like(probs), np.zeros_like(probs)
            for i in range(len(bin_centers_)):
                gids_tmp = gids[np.logical_and((bin_idx[assembly_id] == i + 1),
                                               (cond_df[assembly_id] == key_id).to_numpy())]
                idx = np.in1d(gids_tmp, assembly_gids, assume_unique=True)
                probs[i], probs_low[i], probs_high[i] = prob_with_binom_ci(idx, bin_min_n)
            assembly_probs[key][assembly_id] = probs
            assembly_probs_low[key][assembly_id] = probs_low
            assembly_probs_high[key][assembly_id] = probs_high
    return bin_centers_plot, assembly_probs, assembly_probs_low, assembly_probs_high, chance_levels


def _sign(bin_centers, probs, counts=None):
    """Gets sign of line fitted to assembly probs. vs. indegree"""
    return np.sign(np.polyfit(bin_centers, probs, 1, w=counts)[0])


def assembly_rel_frac_entropy_explained(gids, assembly_grp, bin_centers, bin_idx, seed, bin_min_n, sign_th):
    """Gets mutual information between assembly membership and structural innervation (using pre-binned indegrees)
    and gives it a sign (hence 'relative') based on fitting the probabilities with a line"""
    if isinstance(seed, str) and seed not in ["consensus", "average"]:
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
            for k in range(len(bin_centers[key])):
                tmp = idx[bin_idx_ == k + 1]
                counts[k], probs[k] = len(tmp), np.mean(tmp)
            valid_n_idx = np.where(counts >= bin_min_n)[0]
            if len(valid_n_idx):
                mi_sign = _sign(bin_centers[key][valid_n_idx], probs[valid_n_idx], counts[valid_n_idx])
                mi_matrix[i, j] = mi_sign * mi
                mi_ctrl_matrix[i, j] = mi_ctrl
            else:
                mi_matrix[i, j], mi_ctrl_matrix[i, j] = np.nan, np.nan
    # set values that are smaller than control mean + significance threshold * control std to nan...
    if sign_th > 0:
        mi_matrix[np.abs(mi_matrix) < (np.nanmean(mi_ctrl_matrix) + sign_th * np.nanstd(mi_ctrl_matrix))] = np.nan
    ratio = (np.nanmean(np.abs(mi_matrix)) - np.nanmean(mi_ctrl_matrix)) / np.nanstd(mi_ctrl_matrix)
    print("MI ratio (between data and shuffled/control data): %.2f" % ratio)
    return pd.DataFrame(data=mi_matrix, columns=assembly_idx, index=keys)


def assembly_cond_frac_entropy_explained(gids, assembly_grp, bin_idx, bin_idx_cond, seed, sign_th):
    """Gets conditional mutual information between assembly membership and structural innervation
    (using pre-binned indegrees). (Unlike above here it would be hard to define the sign, so we just skip it...)"""
    if isinstance(seed, str) and seed not in ["consensus", "average"]:
        seed = int(seed.split("seed")[1])
    keys = np.sort(list(bin_idx.keys()))
    assembly_idx = np.sort([assembly.idx[0] for assembly in assembly_grp.assemblies])
    mi_matrix = np.zeros((len(keys), len(assembly_idx)), dtype=np.float32)
    mi_ctrl_matrix = np.zeros_like(mi_matrix)
    for j, assembly_id in enumerate(assembly_idx):
        for i, key in enumerate(keys):
            idx = np.in1d(gids, assembly_grp.loc((assembly_id, seed)).gids, assume_unique=True)
            bin_idx_ = bin_idx[key].copy()
            bin_idx_cond_ = bin_idx_cond[key]
            mi_matrix[i, j] = drv.information_mutual_conditional(idx, bin_idx_, bin_idx_cond_)\
                              / drv.entropy_conditional(idx, bin_idx_cond_)
            mi_ctrl_matrix[i, j] = drv.information_mutual_conditional(idx, np.random.permutation(bin_idx_), bin_idx_cond_)\
                                   / drv.entropy_conditional(idx, bin_idx_cond_)
    # set values that are smaller than control mean + significance threshold * control std to nan...
    if sign_th > 0:
        mi_matrix[mi_matrix < (np.mean(mi_ctrl_matrix) + sign_th * np.std(mi_ctrl_matrix))] = np.nan
    ratio = (np.nanmean(mi_matrix) - np.mean(mi_ctrl_matrix)) / np.std(mi_ctrl_matrix)
    print("Conditional MI ratio (between data and shuffled/control data): %.2f" % ratio)
    return pd.DataFrame(data=mi_matrix, columns=assembly_idx, index=keys)

