# -*- coding: utf-8 -*-
"""
Advanced network metrics on `connectivity.py`
authors: Daniela Egas Santander, Nicolas Ninin
last modified: 12.2020
"""

import numpy as np

from assemblyfire.connectivity import ConnectivityMatrix


def closeness_connected_components(matrix, directed=False):
    """
    Compute the closeness of each connected component of more than 1 vertex
    :param matrix: shape (n,n)
    :param directed: if True compute using strong component and directed closeness
    :return a list of array of shape n, containting closeness of vertex in this component
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
    return all_c


class NetworkAssembly(ConnectivityMatrix):
    """
    A class derived from ConnectivityMatrix with additional information on networks metrics
    of the subgraph associated to an assembly within the connectivity matrix of the circuit.
    """

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

    def simplex_counts(self, sub_gids):
        """Return the simplex counts of submatrix specified by `sub_gids`"""
        import pyflagser

        sub_gids = self.__extract_gids__(sub_gids)
        sub_mat = self.submatrix(sub_gids)
        return pyflagser.flagser_count_unweighted(sub_mat, directed=True)

    def betti_counts(self, sub_gids):
        """Return the betti counts of submatrix specified by `sub_gids`"""
        import pyflagser

        sub_gids = self.__extract_gids__(sub_gids)
        sub_mat = self.submatrix(sub_gids)
        return pyflagser.flagser_unweighted(sub_mat, directed=True)

    def convex_hull(self, sub_gids):
        """Return the convex hull of the sub gids in the 3D space. Require to know x,y,z position for gids"""
        pass

    def centrality(self, sub_gids, kind="closeness", directed=False):
        """Compute a centrality for the sub graph defined by sub_gids. `kind` can be 'betweeness' or 'closeness'"""
        if kind == "closeness":
            return self.closeness(sub_gids, directed)
        else:
            ValueError("Kind must be 'closeness'!")

    def communicability(self, sub_gids):
        pass

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
    # or on edges (strength of connection).


def simplex_counts_consensus(consensus_assemblies_dict, circuit):
    """
    Computes the simplices of consensus assemblies and a random control of the size of the average of each instantion.
    :param consensus_assemblies_dict: A dictionary with consensus assemblies.
    :param circuit: A NetworkAssembly object for the circuit where the assemblies belong to.
    :return simplex_count: A dictionary with the same keys as consensus_assemblies_dict
        with values lists of simplex counts for all the instantions of the consensus assembly in each key
    :return simplex_count_control: A dictionary with the same keys
        and values a simplex counts of random controls of size the average size
        of the instantions of each conensus assembly
    """
    # Compute simplex counts for assemblies within clusters
    simplex_count = {}
    simplex_count_control = {}
    for k, consensus_assembly in consensus_assemblies_dict.items():
        # Simplex count of instantions within one cluster (under one key)
        simplex_count[k] = [circuit.simplex_counts(inst.gids) for inst in consensus_assembly.instantiations]
        # Comparison with random control of average size of the instantions
        mean_size = int(np.mean([len(inst.gids) for inst in consensus_assembly.instantiations]))
        sample_gids = np.random.choice(circuit.gids, mean_size, replace=False)
        simplex_count_control[k] = [circuit.simplex_counts(sample_gids)]
    return simplex_count, simplex_count_control


def simplex_counts_union(consensus_assemblies_dict, circuit):
    """
    Computes the simplex counts of the union of the consensus assemblies across consensus_assemblies_dict
    :param consensus_assemblies_dict: A dictionary with consensus assemblies.
    :param circuit: A NetworkAssembly object for the circuit where the assemblies belong to.
    :return simplex_count: A dictionary with the same keys as consensus_assemblies_dict
        with values simplex counts for for the union of the consensus assembly in each key
    :return simplex_count_control: A dictionary with the same keys
        and values a simplex counts of random controls of the same size of its corresponding union
    """
    simplex_count = {}
    simplex_count_control = {}
    for k, consensus_assembly in consensus_assemblies_dict.items():
        union = consensus_assembly.union
        simplex_count[k] = [circuit.simplex_counts(union)]
        simplex_count_control[k] = [circuit.simplex_counts(circuit.sample_gids_n_neurons(union))]
    # TODO: Add controls also for mtype and depth
    return simplex_count, simplex_count_control


def simplex_counts_core(consensus_assemblies_dict, circuit):
    """
    Computes the simplex counts of the core of the consensus assemblies across consensus_assemblies_dict
    :param consensus_assemblies_dict: A dictionary with consensus assemblies.
    :param circuit: A NetworkAssembly object for the circuit where the assemblies belong to.
    :return simplex_count: A dictionary with the same keys as consensus_assemblies_dict
        with values simplex counts for for the core of the consensus assembly in each key
    :return simplex_count_control: A dictionary with the same keys
        with values  simplex counts of random controls of the same size of its corresponding core
    """
    simplex_count = {}
    simplex_count_control = {}
    for k, consensus_assembly in consensus_assemblies_dict.items():
        simplex_count[k] = [circuit.simplex_counts(consensus_assembly)]
        simplex_count_control[k] = [circuit.simplex_counts(circuit.sample_gids_n_neurons(
            consensus_assembly, consensus_assembly.union))]
    # TODO: Add controls also for mtype and depth
    return simplex_count, simplex_count_control


def simplex_counts_intersection(consensus_assemblies_dict, circuit):
    """
    Computes the simplex counts of the intersection of the consensus assemblies across consensus_assemblies_dict
    :param consensus_assemblies_dict: A dictionary with consensus assemblies.
    :param circuit: A NetworkAssembly object for the circuit where the assemblies belong to.
    :return simplex_count: A dictionary with the same keys as consensus_assemblies_dict
        with values simplex counts for the intersection of the consensus assembly in each key
    :return simplex_count_control: A dictionary with the same keys
        and values a simplex counts of random controls of the same size of its corresponding intersection
    """
    simplex_count = {}
    simplex_count_control = {}
    for k, consensus_assembly in consensus_assemblies_dict.items():
        max_filtration = np.max(consensus_assembly.coreness)
        gids_intersection = consensus_assembly.union.gids[np.where(consensus_assembly.coreness == max_filtration)]
        # TODO: Implement the above in a weighted assembly class where you can choose thresholds
        simplex_count[k] = [circuit.simplex_counts(gids_intersection)]
        simplex_count_control[k] = [circuit.simplex_counts(circuit.sample_gids_n_neurons(
            gids_intersection, consensus_assembly.union))]
    # TODO: Add controls also for mtype and depth
    return simplex_count, simplex_count_control


def simplex_counts_core_vs_intersection(consensus_assemblies_dict, circuit):
    """Computes the simplex counts of the core and intersection of the consensus assemblies across consensus_assemblies_dict
        :param consensus_assemblies_dict: A dictionary with consensus assemblies.
        :param circuit: A NetworkAssembly object for the circuit where the assemblies belong to.
        :return simplex_count_core:
        :return simplex_count_intersection:
    """
    simplex_count_core = {}
    simplex_count_intersection = {}
    for k, consensus_assembly in consensus_assemblies_dict.items():
        simplex_count_core[k] = [circuit.simplex_counts(consensus_assembly)]
        max_filtration = np.max(consensus_assembly.coreness)
        gids_intersection = consensus_assembly.union.gids[np.where(consensus_assembly.coreness == max_filtration)]
        # TODO: Implement the above in a weighted assembly class where you can choose thresholds
        simplex_count_intersection[k] = [circuit.simplex_counts(gids_intersection)]
    # TODO: Add controls also for mtype and depth
    return simplex_count_core, simplex_count_intersection
