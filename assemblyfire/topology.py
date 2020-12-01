from assemblyfire.connectivity import ConnectivityMatrix

#Currently not working because ConnectivityMatrix has changed!
class NetworkAssembly(ConnectivityMatrix):
    """
    A class derived from ConnectivityMatrix with additional information on networks metrics of the subgraph associated to an assembly within the connectivity matrix of the circuit.
    """
    
    
    def simplex_counts(self, sub_gids):
        """Return a the simplex counts of submatrix specified by `sub_gids`"""
        import pyflagser
        sub_mat = self.submatrix(self, sub_gids)
        return pyflagser.flagser_count_unweighted(sub_mat, directed=True)
        

    def betti_counts(self,sub_gids):
        """Return a the betti counts of submatrix specified by `sub_gids`"""
        import pyflagser
        sub_mat= self.submatrix(self, sub_gids)
        return pyflagser.flagser_unweighted(sub_mat, directed=True)

    def convex_hull(self, sub_gids):
        """
        Return the convex hull of the sub gids in the 3D space.
        Require to know x,y,z position for gids
        """
        pass
    def centrality(self, sub_gids,kind = "betweeness"):
        """
        compute a centrality for the sub graph defined by sub_gids
        kind can be betweeness, closeness
        """
        pass
    def connected_components(self, sub_gids = None):
        """
        compute connected_components of the subgraph, if None compute on the whole graph
        """
        if sub_gids != None:
            sub_mat = self.submatrix(self,sub_gids)
        else:
            sub_mat = self._m
        from scipy.sparse.csgraph import connected_components
        pass

    def core_number(self,sub_gids):
        """Returns k core of directed graph, where degree of a vertex is the sum of in degree and out degree"""
        #TODO: Implement directed (k,l) core and k-core of underlying undirected graph (very similar to this)
        import networkx
        G = networkx.from_numpy_matrix(self)
        return networkx.algorithms.core.core_number(G) #Very inefficient (returns a dictionary!).  Look for different implementation
        
    
    #TODO: Simplex counts associated to a dictionary of assembly groups e.g. consensus assembly
    #TODO: Simplex counts of core vs. random controls
    #TODO: Filtered simplex counts with different weights on vertices (coreness, intersection) or on edges (strength of connection).
    #TODO: Other graph metrics.  Centrality, communicability, connected components
    
    #Old code from Michael.  I think we don't need this anymore, because we inherit from ConnectivityMatrix, but I keep it here for now. --Daniela
    """def __init__(self, assemblies, all_gids, connectivity_obj, label=None, metadata=None):
        self._connectivity = connectivity_obj
        super(NetworkAssemblyGroup, self).__init__(assemblies, all_gids, label=label, metadata=metadata)

    @classmethod
    def attach_connectivity(cls, base_group, connectivity_obj):
        return cls(base_group.assemblies, base_group.all, connectivity_obj,
                   label=base_group.label, metadata=base_group.metadata)

    def mat_of(self, idx):
        #Returns the connection matrix of a contained assembly
        :return:
        
        return self._connectivity.submat(self.iloc(idx))

    def __mul__(self, other):
        base_spec = super(NetworkAssemblyGroup, self).__mul__(other)
        return self.attach_connectivity(base_spec, self._connectivity)

    def __add__(self, other):
        base_spec = super(NetworkAssemblyGroup, self).__add__(other)
        return self.attach_connectivity(base_spec, self._connectivity)

    def to_h5(self, filename, prefix=None, version=None):
        # TODO: Save the connectivity matrix somewhere in the metadata at 'prefix'
        super(NetworkAssemblyGroup, self).to_h5(filename, prefix=prefix, version=version)"""
