# -*- coding: utf-8 -*-
"""
Advanced network metrics on `connectivity.py`
authors: Daniela Egas Santander, Nicolas Ninin, András Ecker
last modified: 12.2020
"""

import numpy as np
from tqdm import tqdm

from assemblyfire.connectivity import ConnectivityMatrix


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
        in_degrees[seed] = {assembly.idx: circuit.degree(assembly, kind="in")
                            for assembly in assembly_grp.assemblies}
        in_degrees_control[seed]["n"] = {assembly.idx: circuit.degree(
                                         circuit.sample_gids_n_neurons(assembly), kind="in")
                                         for assembly in assembly_grp.assemblies}
        in_degrees_control[seed]["depths"] = {assembly.idx: circuit.degree(
                                              circuit.sample_gids_depth_profile(assembly), kind="in")
                                              for assembly in assembly_grp.assemblies}
        in_degrees_control[seed]["mtypes"] = {assembly.idx: circuit.degree(
                                              circuit.sample_gids_mtype_composition(assembly), kind="in")
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
                                             circuit.sample_gids_n_neurons(assembly))
                                             for assembly in assembly_grp.assemblies}
        simplex_counts_control[seed]["depths"] = {assembly.idx: circuit.simplex_counts(
                                                  circuit.sample_gids_depth_profile(assembly))
                                                  for assembly in assembly_grp.assemblies}
        simplex_counts_control[seed]["mtypes"] = {assembly.idx: circuit.simplex_counts(
                                                  circuit.sample_gids_mtype_composition(assembly))
                                                  for assembly in assembly_grp.assemblies}
    return simplex_counts, simplex_counts_control

def generate_controls(circuit, ref_gids, N, sub_gids=None, control_type="n_neurons"):
    """Generates random controls for ref_gids within sub_gids.  If sub_gids is None,then samples in all of circuit.
        :param N: Number of random controls
        :param control_type:n_neurons, depth, mtype (see sample_gids_ in connectivity.py)"""
    if control_type=="n_neurons":
        return [circuit.sample_gids_n_neurons(ref_gids,sub_gids) for i in range(N)]
    elif control_type=="depth_profile":
        return [circuit.sample_gids_depth_profile(ref_gids,sub_gids) for i in range(N)]
    elif control_type=="mtype_composition":
        return [circuit.sample_gids_mtype_composition(ref_gids,sub_gids) for i in range(N)]


def simplex_counts_consensus(consensus_assemblies_dict, circuit,N):
    """
    Computes the simplices of consensus assemblies and a random control of the size of the average of each instantion.
    :param consensus_assemblies_dict: A dictionary with consensus assemblies.
    :param circuit: A NetworkAssembly object for the circuit where the assemblies belong to.
    :param N: Number of random controls
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
        simplex_count_control[k]=[]
        for i in range(N):
            sample_gids = np.random.choice(circuit.gids, mean_size, replace=False)
            simplex_count_control[k].append(circuit.simplex_counts(sample_gids))
    return simplex_count, simplex_count_control

def simplex_counts_dict(ref_assemblies_dict, circuit,N,sub_assemblies_dict=None):
    """
    Computes the simplex counts of the assemblies in assemblies_dict and random controls.
    :param ref_assemblies_dict: A dictionary with assembly objects.
    :param circuit: A NetworkAssembly object for the circuit where the assemblies belong to.
    :param N: Number of random samples
    :return simplex_count: A dictionary with the same keys as ref_assemblies_dict
        with values simplex counts for the assemblies in each key
    :return simplex_count_control: A dictionary with the same keys
        and values a simplex counts of random controls within sub_assemblies or within circuit if sub_assemblies=None
    """
    simplex_count = {}
    simplex_count_control = {key: {} for key in ["n","depth","mtype"]}
    for k, assembly in ref_assemblies_dict.items():
        simplex_count[k] = [circuit.simplex_counts(assembly)]
        if sub_assemblies_dict==None:
            simplex_count_control["n"][k] = [circuit.simplex_counts(x) for x in generate_controls(circuit, assembly,N)]
            simplex_count_control["depth"][k] = [circuit.simplex_counts(x) for x in generate_controls(circuit, assembly,N, control_type="depth_profile")]
            simplex_count_control["mtype"][k] = [circuit.simplex_counts(x) for x in generate_controls(circuit, assembly,N, control_type="mtype_composition")]
        else:
            assert ref_assemblies_dict.keys()==sub_assemblies_dict.keys(), "Ref assemblies and sub assemblies don't have the same keys"
            sub_gids=sub_assemblies_dict[k]
            simplex_count_control["n"][k] = [circuit.simplex_counts(x) for x in generate_controls(circuit, assembly,N, sub_gids=sub_gids)]
            simplex_count_control["depth"][k] = [circuit.simplex_counts(x) for x in generate_controls(circuit, assembly,N, sub_gids=sub_gids, control_type="depth_profile")]
            simplex_count_control["mtype"][k] = [circuit.simplex_counts(x) for x in generate_controls(circuit, assembly,N, sub_gids=sub_gids, control_type="mtype_composition")]
            
    return simplex_count, simplex_count_control


def simplex_counts_union(consensus_assemblies_dict, circuit,N,sub_assemblies_dict=None):
    """Computes the simplex counts of the union of the consensus assemblies vs. N random controls"""
    ref_assemblies_dict = dict((k, consensus_assembly.union) for k,consensus_assembly in consensus_assemblies_dict.items())
    return simplex_counts_dict(ref_assemblies_dict, circuit,N)
    

def simplex_counts_core(consensus_assemblies_dict, circuit,N, sample_type="union"):
    """Computes the simplex counts of the core of the consensus assemblies vs. N random controls
        :param sample_type: union - generate random control within union, all - generate random control in circuit"""
    ref_assemblies_dict = dict((k, consensus_assembly.gids) for k,consensus_assembly in consensus_assemblies_dict.items())
    sub_assemblies_dict = dict((k, consensus_assembly.union) for k,consensus_assembly in consensus_assemblies_dict.items())
    if sample_type == "union":
        return simplex_counts_dict(ref_assemblies_dict, circuit,N,sub_assemblies_dict=sub_assemblies_dict)
    else:
        assert sample_type == "all", "mode must be either union or all"
        return simplex_counts_dict(ref_assemblies_dict, circuit,N)



def simplex_counts_intersection(consensus_assemblies_dict, circuit,N, sample_type="union"):
    """Computes the simplex counts of the intersection of the consensus assemblies vs. N random controls
        :param sample_type: union - generate random control within union, all - generate random control in circuit"""

    sub_assemblies_dict = dict((k, consensus_assembly.union) for k,consensus_assembly in consensus_assemblies_dict.items())
    ref_assemblies_dict={}
    for k, consensus_assembly in consensus_assemblies_dict.items():
        max_filtration = np.max(consensus_assembly.coreness)
        ref_assemblies_dict[k] = consensus_assembly.union.gids[np.where(consensus_assembly.coreness == max_filtration)]
        # TODO: Implement the above in a weighted assembly class where you can choose thresholds
    if sample_type == "union":
        return simplex_counts_dict(ref_assemblies_dict, circuit,N,sub_assemblies_dict=sub_assemblies_dict)
    else:
        assert sample_type == "all", "mode must be either union or all"
        return simplex_counts_dict(ref_assemblies_dict, circuit,N)
        
        
def simplex_counts_core_vs_intersection(consensus_assemblies_dict, circuit):
    """
    Computes the simplex counts of the core and intersection of the consensus assemblies across consensus_assemblies_dict
    :param consensus_assemblies_dict: A dictionary with consensus assemblies.
    :param circuit: A NetworkAssembly object for the circuit where the assemblies belong to.
    :return simplex_count_core: A dictionary with the same keys as consensus_assemblies_dict
        with values simplex counts for for the core of the consensus assembly in each key
    :return simplex_count_intersection: A dictionary with the same keys as consensus_assemblies_dict
        with values simplex counts for the intersection of the consensus assembly in each key
    """
    simplex_count_core = {}
    simplex_count_intersection = {}
    for k, consensus_assembly in consensus_assemblies_dict.items():
        simplex_count_core[k] = [circuit.simplex_counts(consensus_assembly)]
        max_filtration = np.max(consensus_assembly.coreness)
        gids_intersection = consensus_assembly.union.gids[np.where(consensus_assembly.coreness == max_filtration)]
        # TODO: Implement the above in a weighted assembly class where you can choose thresholds
        simplex_count_intersection[k] = [circuit.simplex_counts(gids_intersection)]
    # TODO: Add controls maybe? --> Better add filtered version
    return simplex_count_core, simplex_count_intersection
