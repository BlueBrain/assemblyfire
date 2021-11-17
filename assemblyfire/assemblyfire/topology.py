# -*- coding: utf-8 -*-
"""
Advanced network metrics on `connectivity.py`
authors: Daniela Egas Santander, Nicolas Ninin, András Ecker
last modified: 12.2020
"""

import numpy as np
from tqdm import tqdm

from assemblyfire.connectivity import ConnectivityMatrix
# from assemblyfire.assemblies import WeightedAssembly, AssemblyGroup


def closeness_connected_components(matrix, directed=False, return_sum=True):
    """
    Compute the closeness of each connected component of more than 1 vertex
    :param matrix: shape (n,n)
    :param directed: if True compute using strong component and directed closeness
    :param return_sum: if True return only one list given by summing over all the component
    :return a signle array( if return_sum=True) or a list of array of shape n,
    containting closeness of vertex in this component
    or 0 if vertex is not in the component, in any case closeness cant be zero otherwise
    """
    from sknetwork.ranking import Closeness
    from scipy.sparse.csgraph import connected_components

    if directed:
        n_comp, comp = connected_components(matrix, directed=True, connection="strong")
    else:
        n_comp, comp = connected_components(matrix, directed=False)
        matrix = matrix + matrix.T  # we need to make the matrix symmetric

    closeness = Closeness()  # if matrix is not symmetric automatically use directed
    n = matrix.shape[0]
    all_c = []
    for i in range(n_comp):
        c = np.zeros(n)
        idx = np.where(comp == i)[0]
        sub_mat = matrix[np.ix_(idx, idx)].tocsr()
        if sub_mat.getnnz() > 0:
            c[idx] = closeness.fit_transform(sub_mat)
            all_c.append(c)
    if return_sum:
        all_c = np.array(all_c)
        return np.sum(all_c, axis=0)
    else:
        return all_c


class NetworkAssembly(ConnectivityMatrix):
    """
    A class derived from ConnectivityMatrix with additional information on networks metrics
    of the subgraph associated to an assembly within the connectivity matrix of the circuit.
    """
    def __extract_gids__(self, sub_gids):
        return self.__extract_vertex_ids__(sub_gids)

    def degree(self, sub_gids=None, kind="in"):
        """Return in/out degrees of the subgraph, if None compute on the whole graph"""
        if sub_gids is not None:
            matrix = self.subarray(self.__extract_gids__(sub_gids))
        else:
            matrix = self.array
        if kind == "in":
            return np.sum(matrix, axis=0)
        elif kind == "out":
            return np.sum(matrix, axis=1)
        else:
            ValueError("Need to specify 'in' or 'out' degree!")

    def density(self, sub_gids=None):
        if sub_gids is None:
            m = self.matrix
        else:
            m = self.submatrix(sub_gids)
        return m.getnnz()/np.prod(m.shape)

    def simplex_counts(self, sub_gids):
        """Return the simplex counts of submatrix specified by `sub_gids`"""
        import pyflagser
        sub_mat = self.submatrix(self.__extract_gids__(sub_gids))
        return pyflagser.flagser_count_unweighted(sub_mat, directed=True)

    def betti_counts(self, sub_gids):
        """Return the betti counts of submatrix specified by `sub_gids`"""
        import pyflagser
        sub_mat = self.submatrix(self.__extract_gids__(sub_gids))
        return pyflagser.flagser_unweighted(sub_mat, directed=True)["betti"]

    def convex_hull(self, sub_gids):
        """Return the convex hull of the sub gids in the 3D space. Require to know x,y,z position for gids"""
        pass

    def communicability(self, sub_gids):
        pass

    def centrality(self, sub_gids, kind="closeness", directed=False):
        """Compute a centrality for the sub graph defined by sub_gids. `kind` can be 'betweeness' or 'closeness'"""
        if kind == "closeness":
            return self.closeness(sub_gids, directed)
        else:
            ValueError("Kind must be 'closeness'!")

    def closeness(self, sub_gids=None, directed=False):
        """Compute closeness centrality using sknetwork on all connected components or strongly connected
        component (if directed==True)"""
        if sub_gids is not None:
            m = self.submatrix(self.__extract_gids__(sub_gids))
        else:
            m = self.matrix
        return closeness_connected_components(m, directed=directed)

    def connected_components(self, sub_gids=None):
        """Returns a list of the size of the connected components of the underlying undirected graph on sub_gids,
        if None, compute on the whole graph"""
        import networkx as nx

        if sub_gids is not None:
            matrix = self.subarray(self.__extract_gids__(sub_gids))
        else:
            matrix = self.array

        """I keep it here for now in case it's faster to implement it with scipy -- Daniela
        from scipy.sparse.csgraph import connected_components

        if sub_gids is not None:
            sub_mat = self.submatrix(self.__extract_gids__(sub_gids))
        else:
            sub_mat = self.matrix"""

        matrix_und = np.where((matrix+matrix.T) >= 1, 1, 0)
        # TODO: Change the code from below to scipy implementation!
        G = nx.from_numpy_matrix(matrix_und)
        return [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
        # TODO:  Possibly change this to list of gids for each connecte component for this use the line below
        # return sorted(nx.connected_components(G) # Is this coming out in a usable way or should we transform to gids?

    def core_number(self, sub_gids):
        """Returns k core of directed graph, where degree of a vertex is the sum of in degree and out degree"""
        # TODO: Implement directed (k,l) core and k-core of underlying undirected graph (very similar to this)
        import networkx
        G = networkx.from_numpy_matrix(self.submatrix(self.__extract_gids__(sub_gids)))
        # Very inefficient (returns a dictionary!). TODO: Look for different implementation
        return networkx.algorithms.core.core_number(G)

    # TODO: Filtered simplex counts with different weights on vertices (coreness, intersection)
    #  or on edges (strength of connection).


def in_degree_assemblies(assembly_grp_dict, circuit):
    """
    Computes the indegree distribution of assemblies across seeds
    and a random controls of the same size/depth profile/mtype composition
    :param assembly_grp_dict: dict with seeds as keys and assembly groups as values
    :param circuit: NetworkAssembly object for the circuit where the assemblies belong to
    :return in_degrees: dict with the same keys as `assembly_grp_dict` - within that an other dict
        with keys as assembly labels and list of in degrees as values
    :return in_degrees_control: dict with the same keys as `assembly_grp_dict` - within that an other dict
        with keys ['n', 'depths', 'mtype'] and yet an other dict similar to `in_degrees` but values are simplex
        counts of the random controls
    """
    in_degrees = {}
    in_degrees_control = {seed: {} for seed in list(assembly_grp_dict.keys())}
    for seed, assembly_grp in assembly_grp_dict.items():
        in_degrees[seed] = {assembly.idx: circuit.degree(assembly, kind="in") for assembly in assembly_grp.assemblies}
        in_degrees_control[seed]["n"] = {assembly.idx: circuit.degree(
                                         circuit.sample_vertices_n_neurons(assembly), kind="in")
                                         for assembly in assembly_grp.assemblies}
        in_degrees_control[seed]["depths"] = {assembly.idx: circuit.degree(
                                              circuit.sample_vertices_from_numerical_property(assembly), kind="in")
                                              for assembly in assembly_grp.assemblies}
        in_degrees_control[seed]["mtypes"] = {assembly.idx: circuit.degree(
                                              circuit.sample_vertices_from_categorical_property(assembly), kind="in")
                                              for assembly in assembly_grp.assemblies}
    return in_degrees, in_degrees_control


def simplex_counts_assemblies(assembly_grp_dict, circuit):
    """
    Computes the simplices of assemblies across seeds
    and a random controls of the same size/depth profile/mtype composition
    :param assembly_grp_dict: dict with seeds as keys and assembly groups as values
    :param circuit: NetworkAssembly object for the circuit where the assemblies belong to
    :return simplex_count: dict with the same keys as `assembly_grp_dict` - within that an other dict
        with keys as assembly labels and list of simplex counts as values
    :return simplex_counts_control: dict with the same keys as `assembly_grp_dict` - within that an other dict
        with keys ['n', 'depths', 'mtype'] and yet an other dict similar to `simplex_count` but values are simplex
        counts of the random controls
    """
    simplex_counts = {}
    simplex_counts_control = {seed: {} for seed in list(assembly_grp_dict.keys())}
    for seed, assembly_grp in tqdm(assembly_grp_dict.items(), desc="Counting simplices"):
        simplex_counts[seed] = {assembly.idx: circuit.simplex_counts(assembly)
                                for assembly in assembly_grp.assemblies}
        simplex_counts_control[seed]["n"] = {assembly.idx: circuit.simplex_counts(
                                             circuit.sample_vertices_n_neurons(assembly))
                                             for assembly in assembly_grp.assemblies}
        simplex_counts_control[seed]["depths"] = {assembly.idx: circuit.simplex_counts(
                                                  circuit.sample_vertices_from_numerical_property(assembly))
                                                  for assembly in assembly_grp.assemblies}
        simplex_counts_control[seed]["mtypes"] = {assembly.idx: circuit.simplex_counts(
                                                  circuit.sample_vertices_from_categorical_property(assembly))
                                                  for assembly in assembly_grp.assemblies}
    return simplex_counts, simplex_counts_control


def filtered_simplex_counts(weighted_assembly, circuit, method="strength"):
    filtration = weighted_assembly.filtration(method=method)
    simplex_counts = []
    for i in range(len(filtration.assemblies)):
        simplex_counts.append(circuit.simplex_counts(filtration.assemblies[i]))
    return simplex_counts


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


def generate_controls(circuit, ref_gids, N, sub_gids=None, control_type="n_neurons"):
    """Generates `N` random controls of `control_type` for `ref_gids` within `sub_gids`.
    If `sub_gids=None`, then sample from the whole circuit."""
    if control_type == "n_neurons":
        return [circuit.sample_gids_n_neurons(ref_gids, sub_gids) for _ in range(N)]
    elif control_type == "depth_profile":
        return [circuit.sample_gids_depth_profile(ref_gids, sub_gids) for _ in range(N)]
    elif control_type == "mtype_composition":
        return [circuit.sample_gids_mtype_composition(ref_gids, sub_gids) for _ in range(N)]
    else:
        raise ValueError("control_type has to be either 'n_neuron' or 'depth_profile' or 'mtype_composition'!")


def simplex_counts_dict(ref_assemblies_dict, circuit, N, sub_assemblies_dict=None):
    """
    Computes the simplex counts of the assemblies and thier random controls.
    :param ref_assemblies_dict: dictionary with Assembly objects
    :param circuit: NetworkAssembly object of the circuit where the assemblies belong to
    :param N: number of random samples from each kind
    :param sub_assemblies_dict: dictionary with (bigger) Assembly objects. E.g. union of a ConsensusAssembly
    :return simplex_count: dictionary with the same keys as ref_assemblies_dict
        and simplex counts for the assemblies as values
    :return simplex_count_control: A dictionary with keys of random control names. Within those an other dictionary
        with the same keys as ref_assemblies_dict and simplex counts of random controls (within sub_assemblies
        or within the whole circuit if 'sub_assemblies=None') as values
    """
    simplex_count = {}
    simplex_count_control = {key: {} for key in ["n", "depth", "mtype"]}
    for k, assembly in tqdm(ref_assemblies_dict.items(), desc="Counting simplices"):
        simplex_count[k] = [circuit.simplex_counts(assembly)]
        if sub_assemblies_dict is not None:
            assert list(ref_assemblies_dict.keys()) == list(sub_assemblies_dict.keys()),\
                "Ref assemblies and sub assemblies don't have the same keys!"
            simplex_count_control["n"][k] = [circuit.simplex_counts(ctrl) for ctrl in generate_controls(
                                             circuit, assembly, N, control_type="n_neurons",
                                             sub_gids=sub_assemblies_dict[k])]
            simplex_count_control["depth"][k] = [circuit.simplex_counts(ctrl) for ctrl in generate_controls(
                                                 circuit, assembly, N, control_type="depth_profile",
                                                 sub_gids=sub_assemblies_dict[k])]
            simplex_count_control["mtype"][k] = [circuit.simplex_counts(ctrl) for ctrl in generate_controls(
                                                 circuit, assembly, N, control_type="mtype_composition",
                                                 sub_gids=sub_assemblies_dict[k])]
        else:
            simplex_count_control["n"][k] = [circuit.simplex_counts(ctrl) for ctrl in generate_controls(
                                             circuit, assembly, N, control_type="n_neurons")]
            simplex_count_control["depth"][k] = [circuit.simplex_counts(ctrl) for ctrl in generate_controls(
                                                 circuit, assembly, N, control_type="depth_profile")]
            simplex_count_control["mtype"][k] = [circuit.simplex_counts(ctrl) for ctrl in generate_controls(
                                                 circuit, assembly, N, control_type="mtype_composition")]
    return simplex_count, simplex_count_control


def simplex_counts_consensus(consensus_assemblies_dict, circuit, n_ctrls=1):
    """ Computes the simplices of consensus assemblies and a random control of the size of the average of instantiations
    Cannot be done in the same style as the above ones, but the principle is the same"""
    simplex_count = {}
    simplex_count_control = {k: [] for k in list(consensus_assemblies_dict.keys())}
    for k, consensus_assembly in tqdm(consensus_assemblies_dict.items(), desc="Counting simplices"):
        # Simplex count of instantiations within one cluster (under one key)
        simplex_count[k] = [circuit.simplex_counts(inst.gids) for inst in consensus_assembly.instantiations]
        # Comparison with random control of average size of the instantiations
        mean_size = int(np.mean([len(inst.gids) for inst in consensus_assembly.instantiations]))
        for _ in range(n_ctrls):
            sample_gids = np.random.choice(circuit.gids, mean_size, replace=False)
            simplex_count_control[k].append(circuit.simplex_counts(sample_gids))
    return simplex_count, simplex_count_control


def simplex_counts_union(consensus_assemblies_dict, circuit, N):
    """Computes the simplex counts of the union of the consensus assemblies vs. N random controls"""
    ref_assemblies_dict = {k: consensus_assembly.union for k, consensus_assembly
                           in consensus_assemblies_dict.items()}
    return simplex_counts_dict(ref_assemblies_dict, circuit, N, sub_assemblies_dict=None)


def simplex_counts_core(consensus_assemblies_dict, circuit, N, sample_type="union"):
    """Computes the simplex counts of the core of the consensus assemblies vs. N random controls"""
    if sample_type == "all":
        return simplex_counts_dict(consensus_assemblies_dict, circuit, N, sub_assemblies_dict=None)
    elif sample_type == "union":
        sub_assemblies_dict = {k: consensus_assembly.union for k, consensus_assembly
                               in consensus_assemblies_dict.items()}
        return simplex_counts_dict(consensus_assemblies_dict, circuit, N, sub_assemblies_dict=sub_assemblies_dict)
    else:
        raise ValueError("Sample type has to be either 'union' or 'all'!")


def simplex_counts_intersection(consensus_assemblies_dict, circuit, N, sample_type="union"):
    """Computes the simplex counts of the intersection of the consensus assemblies vs. N random controls"""
    intersection_gids_dict = get_intersection_gids(consensus_assemblies_dict)
    if sample_type == "all":
        return simplex_counts_dict(intersection_gids_dict, circuit, N, sub_assemblies_dict=None)
    elif sample_type == "union":
        sub_assemblies_dict = {k: consensus_assembly.union for k, consensus_assembly
                               in consensus_assemblies_dict.items()}
        return simplex_counts_dict(intersection_gids_dict, circuit, N, sub_assemblies_dict=sub_assemblies_dict)
    else:
        raise ValueError("Sample type has to be either 'union' or 'all'!")


def simplex_counts_core_vs_intersection(consensus_assemblies_dict, circuit):
    """Computes the simplex counts of the core and intersection of the consensus assemblies
     in the consensus_assemblies_dict"""
    intersection_gids_dict = get_intersection_gids(consensus_assemblies_dict)
    simplex_count_core = {}
    simplex_count_intersection = {}
    for k, consensus_assembly in tqdm(consensus_assemblies_dict.items(), desc="Counting simplices"):
        simplex_count_core[k] = [circuit.simplex_counts(consensus_assembly)]
        simplex_count_intersection[k] = [circuit.simplex_counts(intersection_gids_dict[k])]
    return simplex_count_core, simplex_count_intersection

