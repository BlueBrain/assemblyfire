# -*- coding: utf-8 -*-
"""
Advanced network metrics on `connectivity.py`
authors: Daniela Egas Santander, Nicolas Ninin
last modified: 12.2020
"""

import numpy as np

from assemblyfire.connectivity import ConnectivityMatrix

class NetworkAssembly(ConnectivityMatrix):
    """
    A class derived from ConnectivityMatrix with additional information on networks metrics
    of the subgraph associated to an assembly within the connectivity matrix of the circuit.
    """

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

    def centrality(self, sub_gids, kind="closeness",directed=False):
        """Compute a centrality for the sub graph defined by sub_gids. `kind` can be 'betweeness' or 'closeness'"""
        if kind =="closenes":
            return self.closeness(sub_gids,directed)
        else:
            print("specify a type of centrality : closeness,???,???")
    def closeness(self,sub_gids=None,directed=False):
        """ compute closeness centrality using sknetwork on all connected components or strongly connected
        component (if directed==True)

        output
        """
        if sub_gids is not None:
            m = self.submatrix(self.__extract_gids__(sub_gids))
        else:
            m = self.matrix
        return closeness_connected_components(m,directed=directed)


    def degree(self, sub_gids=None, kind="in"):
        """Return in/out degrees of the subgraph, if None compute on the whole graph """
        if sub_gids is not None:
            m = self.subarray(self.__extract_gids__(sub_gids))
        else:
            m = self.array

        if kind == "in":
            return np.sum(m, axis=0)
        elif kind == "out":
            return np.sum(m, axis=1)
        else:
            ValueError("Need to specify 'in' or 'out' degree!")


    def connected_components(self, sub_gids=None):
        """Returns a list of the size of the connected components of the underlying undirected graph on sub_gids.
            If sub_gids == None, returns it uses the whole graph"""
                
        import networkx as nx
        if sub_gids==None:
            matrix=np.array(self.dense_matrix)
        else:
            sub_gids = self.__extract_gids__(sub_gids)
            matrix=self.subarray(sub_gids)
        matrix_und=np.where((matrix+matrix.T) >=1, 1, 0)
        
        #Change the code from below to scipy implementation.
        G = nx.from_numpy_matrix(matrix_und)
        return [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
        #TODO:  Possibly change this to list of gids for each connecte component for this use the line below
        #return sorted(nx.connected_components(G) #Is this coming out in a usable way or should we transform to gids?


        #OLD CODE.  I keep it here for now in case it's faster to implement it with scipy --Daniela
        """from scipy.sparse.csgraph import connected_components

        if sub_gids is not None:
            sub_gids = self.__extract_gids__(sub_gids)
            sub_mat = self.submatrix(sub_gids)
        else:
            sub_mat = self.matrix"""

    def core_number(self, sub_gids):
        """Returns k core of directed graph, where degree of a vertex is the sum of in degree and out degree"""
        #TODO: Implement directed (k,l) core and k-core of underlying undirected graph (very similar to this)
        import networkx
        sub_gids = self.__extract_gids__(sub_gids)
        G = networkx.from_numpy_matrix(self.submatrix(sub_gids))
        return networkx.algorithms.core.core_number(G) # Very inefficient (returns a dictionary!). Look for different implementation
        


def closeness_connected_components(matrix,directed=False):
    """ compute the closeness of each  connected component of more than 1 vertex
    matrix: shape (n,n)
    directed: if True compute using strong component and directed closeness
    return a list of array of shape n, containting closeness of vertex in this component
    or 0 if vertex is not in the component, in any case closeness cant be zero otherwise

    """
    from sknetwork.ranking import Closeness
    from scipy.sparse.csgraph import connected_components
    if directed:
        n_comp,comp=connected_components(matrix,directed=True,connection="strong")
    else:
        n_comp,comp=connected_components(matrix,directed=False)
        # we need to make the matrix symetric
        matrix=matrix+matrix.T
    closeness=Closeness() #if matrix is not symetric automatically use directed
    n=matrix.shape[0]
    all_c=[]
    for i in range(n_comp):
        c=np.zeros(n)
        idx=comp==i
        sub_mat=matrix[np.ix_(idx,idx)].tocsr()
        if sub_mat.getnnz()>0:
            c[idx]=closeness.fit_transform(sub_mat)
            all_c.append(c)
    return all_c
    #TODO: Simplex counts associated to a dictionary of assembly groups e.g. consensus assembly
    #TODO: Simplex counts of core vs. random controls
    #TODO: Filtered simplex counts with different weights on vertices (coreness, intersection) or on edges (strength of connection).
    #TODO: Other graph metrics.  Centrality, communicability, connected components


#Functions to compute things across seeds.  Maybe these should be methods of a class? --Daniela

def simplex_counts_consensus(consensus_assemblies_dict, circuit):
    """Computes the simplices of consensus assemblies and a random control of the size of the average of each instantion
        :param consensus_assemblies_dict: A dictionary with the different consensus assemblies accross seeds
        :param circuit: A NetworkAssembly object for the circuit where the assemblies belong to
        :return simplex_count_dict: A dictionary with the same keys as consensus_assemblies_dict
                                    with values lists of simplex counts for all the instantions of the consensus assembly in each key
        :return simplex_count_control_dict: A dictionary with the same keys and values a simplex counts of random controls of
                                            size the average size of the instantions of each conensus assembly
    """
    #Compute simplex counts for assemblies within clusters
    simplex_count_dict={}
    simplex_count_control_dict={}

    for c in consensus_assemblies_dict.keys():
        #Simplex count of instantions in c
        simplex_count_dict[c]=[circuit.simplex_counts(x.gids) for x in consensus_assemblies_dict[c].instantiations]
        #Comparison with random control of average size of the instantions
        mean_size=int(np.mean(np.array([len(x.gids) for  x in consensus_assemblies_dict[c].instantiations])))
        sample_gids=np.random.choice(circuit.gids, mean_size, replace=False)
        simplex_count_control_dict[c]=[circuit.simplex_counts(sample_gids)]
    return simplex_count_dict, simplex_count_control_dict
    

