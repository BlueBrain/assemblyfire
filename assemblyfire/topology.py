# -*- coding: utf-8 -*-
"""
Advanced network metrics on `connectivity.py`
authors: Daniela Egas Santander, Nicolas Ninin
last modified: 12.2020
"""

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
        sub_mat= self.submatrix(sub_gids)
        return pyflagser.flagser_unweighted(sub_mat, directed=True)

    def convex_hull(self, sub_gids):
        """
        Return the convex hull of the sub gids in the 3D space.
        Require to know x,y,z position for gids
        """
        pass

    def centrality(self, sub_gids, kind="betweeness"):
        """
        compute a centrality for the sub graph defined by sub_gids
        kind can be betweeness, closeness
        """
        pass

    def connected_components(self, sub_gids=None):
        """
        compute connected_components of the subgraph, if None compute on the whole graph
        """
        from scipy.sparse.csgraph import connected_components
        if sub_gids != None:
            sub_gids = self.__extract_gids__(sub_gids)
            sub_mat = self.submatrix(sub_gids)
        else:
            sub_mat = self.matrix
        pass

    def core_number(self, sub_gids):
        """Returns k core of directed graph, where degree of a vertex is the sum of in degree and out degree"""
        #TODO: Implement directed (k,l) core and k-core of underlying undirected graph (very similar to this)
        import networkx
        G = networkx.from_numpy_matrix(self.submatrix(sub_gids))
        return networkx.algorithms.core.core_number(G) # Very inefficient (returns a dictionary!). Look for different implementation
        
    
    #TODO: Simplex counts associated to a dictionary of assembly groups e.g. consensus assembly
    #TODO: Simplex counts of core vs. random controls
    #TODO: Filtered simplex counts with different weights on vertices (coreness, intersection) or on edges (strength of connection).
    #TODO: Other graph metrics.  Centrality, communicability, connected components

