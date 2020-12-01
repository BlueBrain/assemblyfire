# -*- coding: utf-8 -*-
"""
Class to get, save and load connection matrix and sample submatrices from it
authors: Michael Reimann, András Ecker
last modified: 11.2020
"""

import h5py
from tqdm import tqdm
import numpy as np
import pandas


class ConnectivityMatrix(object):
    """Small utility class to hold a connections matrix and generate submatrices"""
    def __init__(self, adj_matrix, gids, depths, mtypes):
        """Not too intuitive init - please see `from_bluepy()` below"""
        self._m = adj_matrix
        self._gids = gids
        self._depths = depths
        self._mtypes = mtypes
        self._lookup = self.__make_lookup__()

    def __make_lookup__(self):
        return pandas.Series(np.arange(len(self.gids)), index=self.gids)

    @property
    def gids(self):
        return self._gids

    @property
    def depths(self):
        return self._depths

    @property
    def mtypes(self):
        return self._mtypes

    @property
    def matrix(self):
        return self._m

    @property
    def dense_matrix(self):
        return self._m.todense()

    @property
    def array(self):
        return np.array(self.dense_matrix)

    @staticmethod
    def __extract_gids__(an_obj):
        from assemblyfire.assemblies import Assembly
        if isinstance(an_obj, Assembly):
            return an_obj.gids
        return an_obj

    @classmethod
    def from_bluepy(cls, blueconfig_path, gids=None):
        """
        BlueConfig based constructor
        :paramfig_path: path to BlueConfig
        :param gids: array of gids aka. nodes of the graph, if None - all excitatory gids from the circuit are used
        """
        from scipy import sparse
        from assemblyfire.spikes import get_bluepy_simulation
        from assemblyfire.utils import get_depths, get_mtypes

        sim = get_bluepy_simulation(blueconfig_path)
        if gids is None:
            from assemblyfire.utils import get_E_gids
            gids = get_E_gids(sim.circuit, sim.target)
        depths = np.asarray(get_depths(sim.circuit, gids))
        mtypes = np.asarray(get_mtypes(sim.circuit, gids))
        conv = pandas.Series(np.arange(len(gids)), index=gids)
        indptr = [0]
        indices = []
        for gid in tqdm(gids, desc="Building connectivity matrix", miniters=len(gids) / 100):
            aff = conv[np.intersect1d(sim.circuit.connectome.afferent_gids(gid), gids)]
            indices.extend(aff)
            indptr.append(len(indices))
        data = np.ones_like(indices, dtype=bool)
        adj_mat = sparse.csc_matrix((data, indices, indptr), shape=(len(gids), len(gids)))
        return cls(adj_mat, gids, depths, mtypes)

    def submatrix(self, sub_gids, sub_gids_post=None):
        """
        Return a submatrix specified by `sub_gids`
        :param sub_gids: Subpopulation to get the submatrix for. Can be either a list of gids, or an Assembly object
        :param sub_gids_post: (optiona) if specified, defines the postsynaptic population. Else pre- equals postsynaptic
        population
        :return: the adjacency submatrix of the specified population(s).
        """
        if sub_gids_post is not None:
            return self.matrix[np.ix_(self._lookup[self.__extract_gids__(sub_gids)],
                                      self._lookup[self.__extract_gids__(sub_gids_post)])]
        idx = self._lookup[self.__extract_gids__(sub_gids)]
        return self.matrix[np.ix_(idx, idx)]

    def dense_submatrix(self, sub_gids, sub_gids_post=None):
        return self.submatrix(sub_gids, sub_gids_post=sub_gids_post).todense()

    def subarray(self, sub_gids, sub_gids_post=None):
        return np.array(self.dense_submatrix(sub_gids, sub_gids_post=sub_gids_post))

    def sample_matrix_n_neurons(self, ref_gids, sub_gids=None):
        """
        Return a submatrix with the same number of neurons as `ref_gids`
        :param ref_gids: Subpopulation to use as reference for sampling.
                 Can be either a list of gids, or an Assembly object
        :param sub_gids: (optional) if specified, subpopulation to sample from
                         (e.g. union of a ConsensusAssembly vs. the core as ref_gids)
        """

        ref_gids = self.__extract_gids__(ref_gids)
        if sub_gids is not None:
            assert np.isin(sub_gids, self.gids).all(), "Sub gids are not part of the connectivity matrix"
            assert np.isin(ref_gids, sub_gids).all(), "Reference gids are not part of sub gids"
        else:
            sub_gids = self.gids
            assert np.isin(ref_gids, sub_gids).all(), "Reference gids are not part of the connectivity matrix"

        sample_gids = np.random.choice(sub_gids, len(ref_gids), replace=False)
        idx = self._lookup[sample_gids]
        return self.matrix[np.ix_(idx, idx)]

    def dense_sample_n_neurons(self, ref_gids, sub_gids):
        return self.sample_matrix_n_neurons(ref_gids).todense()

    def sample_n_neurons(self, ref_gids, sub_gids):
        return np.array(self.dense_sample_n_neurons(ref_gids))

    def sample_gids_depth_profile(self, ref_gids, sub_gids, n_bins=50):
        """
        Return gids with the same (binned) depth profile as `ref_gids`
        :param ref_gids: Subpopulation to use as reference for sampling.
                         Can be either a list of gids, or an Assembly object
        :param n_bins: number of bins to be used to bin depth values
        """
        ref_gids = self.__extract_gids__(ref_gids)
        assert np.isin(ref_gids, self.gids).all(), "Reference gids are not part of the connectivity matrix"

        ref_depths = self.depths[np.searchsorted(self.gids, ref_gids)]
        hist, bin_edges = np.histogram(ref_depths, bins=n_bins)
        depths_bins = np.digitize(self.depths, bins=bin_edges)
        assert len(hist == len(depths_bins[1:-1]))  # `digitize` returns values below and above the spec. bin_edges
        sample_gids = []
        for i in range(n_bins):
            idx = np.where(depths_bins == i+1)[0]
            assert idx.shape[0] >= hist[i], "Not enough neurons at this depths to sample from"
            sample_gids.extend(np.random.choice(self.gids[idx], hist[i], replace=False).tolist())
        return sample_gids

    def sample_matrix_depth_profile(self, ref_gids, n_bins):
        idx = self._lookup[self.sample_gids_depth_profile(ref_gids, n_bins)]
        return self.matrix[np.ix_(idx, idx)]

    def dense_sample_depth_profile(self, ref_gids, n_bins):
        return self.sample_matrix_depth_profile(ref_gids, n_bins).todense()

    def sample_depth_profile(self, ref_gids, n_bins):
        return np.array(self.dense_sample_depth_profile(ref_gids, n_bins))

    def sample_gids_mtype_composition(self, ref_gids):
        """
        Return gids with the same mtype composition as `ref_gids`
        :param ref_gids: Subpopulation to use as reference for sampling.
                         Can be either a list of gids, or an Assembly object
        """
        ref_gids = self.__extract_gids__(ref_gids)
        assert np.isin(ref_gids, self.gids).all(), "Reference gids are not part of the connectivity matrix"

        ref_mtypes = self.mtypes[np.searchsorted(self.gids, ref_gids)]
        mtypes_lst, counts = np.unique(ref_mtypes, return_counts=True)
        sample_gids = []
        for i, mtype in enumerate(mtypes_lst):
            idx = np.where(self.mtypes == mtype)[0]
            assert idx.shape[0] >= counts[i], "Not enough %s to sample from" % mtype
            sample_gids.extend(np.random.choice(self.gids[idx], counts[i], replace=False).tolist())
        return sample_gids

    def sample_matrix_mtype_composition(self, ref_gids):
        idx = self._lookup[self.sample_gids_mtype_composition(ref_gids)]
        return self.matrix[np.ix_(idx, idx)]

    def dense_sample_mtype_composition(self, ref_gids):
        return self.sample_matrix_mtype_composition(ref_gids).todense()

    def sample_mtype_composition(self, ref_gids):
        return np.array(self.dense_sample_mtype_composition(ref_gids))

    @classmethod
    def from_h5(cls, fn, group_name=None, prefix=None):
        from scipy import sparse
        if prefix is None:
            prefix = "connectivity"
        if group_name is None:
            group_name = "full_matrix"
        with h5py.File(fn, 'r') as h5:
            prefix_grp = h5[prefix]
            data_grp = prefix_grp[group_name]
            data = data_grp["data"][:]
            indices = data_grp["indices"][:]
            indptr = data_grp["indptr"][:]
            gids = data_grp["gids"][:]
            depths = data_grp["depths"][:]
            mtypes_ascii = data_grp["mtypes"][:]
        adj_mat = sparse.csc_matrix((data, indices, indptr), shape=(len(gids), len(gids)))
        mtypes = np.array([s.decode("utf-8") for s in mtypes_ascii])
        return cls(adj_mat, gids, depths, mtypes)

    def to_h5(self, fn, group_name=None, prefix=None):
        mtypes_ascii = np.array([s.encode("ascii", "ignore") for s in self.mtypes])
        if prefix is None:
            prefix = "connectivity"
        if group_name is None:
            prefix = "full_matrix"
        with h5py.File(fn, "a") as h5:
            prefix_grp = h5.require_group(prefix)
            data_grp = prefix_grp.create_group(group_name)
            data_grp.create_dataset("data", data=self.matrix.data)
            data_grp.create_dataset("indices", data=self.matrix.indices)
            data_grp.create_dataset("indptr", data=self.matrix.indptr)
            data_grp.create_dataset("gids", data=self.gids)
            data_grp.create_dataset("depths", data=self.depths)
            data_grp.create_dataset("mtypes", data=mtypes_ascii)