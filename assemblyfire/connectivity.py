import numpy
import pandas
from ..assemblies import Assembly


class ConnectivityMatrix(object):
    """
    Small utility class to hold a connections matrix and generate submatrices
    """
    def __init__(self, adj_matrix, gids):
        self._m = adj_matrix
        self._gids = gids
        self._lookup = self.__make_lookup__()

    def __make_lookup__(self):
        return pandas.Series(numpy.arange(len(self._gids)), index=self._gids)

    @staticmethod
    def __extract_gids__(an_obj):
        if isinstance(an_obj, Assembly):
            return an_obj.gids
        return an_obj

    def submat(self, sub_gids, sub_gids_post=None):
        """
        Return a submatrix
        :param sub_gids: Subpopulation to get the submatrix for. Can be either a list of gids, or an Assembly object
        :param sub_gids_post: (optiona) if specified, defines the postsynaptic population. Else pre- equals postsynaptic
        population
        :return: the adjacency submatrix of the specified population(s).
        """
        if sub_gids_post is not None:
            return self._m[numpy.ix_(self._lookup[self.__extract_gids__(sub_gids)],
                                     self._lookup[self.__extract_gids__(sub_gids_post)])]
        idx = self._lookup[self.__extract_gids__(sub_gids)]
        return self._m[numpy.ix_(idx, idx)]

    @classmethod
    def from_bluepy(cls, cfg, gids):
        raise NotImplemented()

    @classmethod
    def from_h5(cls, fn, gids):
        raise NotImplemented()

