from assemblyfire._topological_analysis import AnalysisImplementation


class Bettis(AnalysisImplementation):
    def __init__(self, *args, **kwargs):
        super(Bettis, self).__init__(*args, **kwargs)

    @staticmethod
    def _run(conn):
        import pyflagser
        adj_mat = conn.matrix
        return pyflagser.flagser_unweighted(adj_mat, directed=True)["betti"]


class ClosenessConnectedComponents(AnalysisImplementation):

    @staticmethod
    def _run(conn, directed=False, return_sum=True):
        """
        Compute the closeness of each connected component of more than 1 vertex
        :param conn: ConnectivityMatrix
        :param directed: if True compute using strong component and directed closeness
        :param return_sum: if True return only one list given by summing over all the component
        :return a signle array( if return_sum=True) or a list of array of shape n,
        containting closeness of vertex in this component
        or 0 if vertex is not in the component, in any case closeness cant be zero otherwise
        """
        matrix = conn.matrix.tocsr()
        import numpy as np
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


class Degree(AnalysisImplementation):

    @staticmethod
    def _run(conn, kind='in', weighted=False):
        import numpy
        bins = numpy.arange(conn._shape[0] + 1)
        hist_args = {'bins': bins}
        if weighted:
            hist_args['weights'] = conn._edges[conn._default_edge]

        if kind == 'in':
            return numpy.histogram(conn._edges['col'], **hist_args)[0]
        elif kind == 'out':
            return numpy.histogram(conn._edges['row'], **hist_args)[0]
        elif kind == 'total':
            return numpy.histogram(conn._edges['col'], **hist_args)[0] + \
                   numpy.histogram(conn._edges['row'], **hist_args)[0]
        else:
            raise ValueError("kind must be in ['in', 'out', 'total']")


class SimplexCounts(AnalysisImplementation):

    @staticmethod
    def _run(conn):
        import pyflagser
        return pyflagser.flagser_count_unweighted(conn.matrix, directed=True)


class Centrality(AnalysisImplementation):

    @staticmethod
    def _run(conn, directed=False, return_sum=True, kind='closeness'):
        if kind == "closeness":
            return ClosenessConnectedComponents._run(conn, directed=directed, return_sum=return_sum)
        else:
            raise ValueError("kind must be in ['closeness']!")


class SizeConnectedComponents(AnalysisImplementation):

    @staticmethod
    def _run(conn):
        import networkx as nx
        matrix = conn.matrix.tocoo()
        matrix = (matrix + matrix.T) > 0
        G = nx.from_scipy_sparse_matrix(matrix)
        return [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]


class CoreNumber(AnalysisImplementation):

    @staticmethod
    def _run(conn):
        import networkx
        G = networkx.from_scipy_sparse_matrix(conn.matrix)
        # Very inefficient (returns a dictionary!). TODO: Look for different implementation
        return networkx.algorithms.core.core_number(G)
