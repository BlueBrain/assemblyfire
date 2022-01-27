"""
Advanced network metrics on ConnectivityMatrix (now in `conntility`)
authors: Daniela Egas Santander, Nicolas Ninin, András Ecker
last modified: 01.2022
"""

import numpy as np
from tqdm import tqdm
from conntility.connectivity import ConnectivityMatrix


class AssemblyTopology(ConnectivityMatrix):
    """
    A class derived from ConnectivityMatrix with additional information on networks metrics
    of the subgraph associated to an assembly within the connectivity matrix of the circuit.
    """

    def degree(self, pre_gids=None, post_gids=None, kind="in"):
        """Return in/out degrees of the (symmetric) subarray specified by `pre_gids`
        (if `post_gids` is given as well, then the subarray will be asymmetric)"""
        if pre_gids is not None:
            array = self.subarray(pre_gids, sub_gids_post=post_gids)
        else:
            array = self.array
        if kind == "in":
            return np.sum(array, axis=0)
        elif kind == "out":
            return np.sum(array, axis=1)
        else:
            ValueError("Need to specify 'in' or 'out' degree!")

    def density(self, sub_gids=None):
        """Return the density of submatrix specified by `sub_gids`"""
        if sub_gids is None:
            matrix = self.matrix
        else:
            matrix = self.submatrix(sub_gids)
        return matrix.getnnz()/np.prod(matrix.shape)

    def simplex_counts(self, sub_gids):
        """Return the simplex counts of submatrix specified by `sub_gids`"""
        from pyflagser import flagser_count_unweighted
        sub_mat = self.submatrix(sub_gids)
        return flagser_count_unweighted(sub_mat, directed=True)

    def betti_counts(self, sub_gids):
        """Return the betti counts of submatrix specified by `sub_gids`"""
        from pyflagser import flagser_unweighted
        sub_mat = self.submatrix(sub_gids)
        return flagser_unweighted(sub_mat, directed=True)["betti"]


def in_degree_assemblies(assembly_grp_dict, conn_mat, post_id=None):
    """
    Computes the in degree distribution within assemblies (or cross-assemblies if `post_assembly_id` is specified)
    across seeds and a random controls of the same size/depth profile/mtype composition
    :param assembly_grp_dict: dict with seeds as keys and assembly groups as values
    :param conn_mat: AssemblyTopology object for the circuit where the assemblies belong to
    :param post_id: optional ID of postsynaptic assembly for cross-assembly in degrees
    :return in_degrees: dict with the same keys as `assembly_grp_dict` - within that an other dict
                        with keys as assembly labels and list of in degrees as values
    :return in_d_control: dict with the same keys as `assembly_grp_dict` - within that an other dict
                          with keys ['n', 'depths', 'mtype'] and yet an other dict similar to `in_degrees`
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
                                        conn_mat.index("[PH]y").random_numerical_gids(assembly.gids), post_gids,
                                        kind="in") for assembly in assembly_grp.assemblies}
        in_d_control[seed]["mtypes"] = {assembly.idx: conn_mat.degree(
                                        conn_mat.index("mtype").random_categorical_gids(assembly.gids), post_gids,
                                        kind="in") for assembly in assembly_grp.assemblies}
    return in_degrees, in_d_control


def simplex_counts_assemblies(assembly_grp_dict, conn_mat):
    """
    Computes the simplices of assemblies across seeds
    and a random controls of the same size/depth profile/mtype composition
    :param assembly_grp_dict: dict with seeds as keys and assembly groups as values
    :param conn_mat: AssemblyTopology object for the circuit where the assemblies belong to
    :return simplex_count: dict with the same keys as `assembly_grp_dict` - within that an other dict
                           with keys as assembly labels and list of simplex counts as values
    :return simplex_counts_control: dict with the same keys as `assembly_grp_dict` - within that an other dict
                                    with keys ['n', 'depths', 'mtype'] and yet an other dict similar to `simplex_count`
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
                                       conn_mat.index("[PH]y").random_numerical_gids(assembly.gids))
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

