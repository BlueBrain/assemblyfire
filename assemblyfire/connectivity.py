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
    def __init__(self, adj_matrix, gids):
        self._m = adj_matrix
        self._gids = gids
        self._lookup = self.__make_lookup__()

    def __make_lookup__(self):
        return pandas.Series(np.arange(len(self._gids)), index=self._gids)

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
        from assemblyfire.spikes import get_bluepy_simulation
        from scipy import sparse

        sim = get_bluepy_simulation(blueconfig_path)
        if gids is None:
            from assemblyfire.utils import get_E_gids
            gids = get_E_gids(sim.circuit, sim.target)
        conv = pandas.Series(np.arange(len(gids)), index=gids)
        indptr = [0]
        indices = []
        for gid in tqdm(gids, desc="Building connectivity matrix", miniters=len(gids) / 100):
            aff = conv[np.intersect1d(sim.circuit.connectome.afferent_gids(gid), gids)]
            indices.extend(aff)
            indptr.append(len(indices))
        data = np.ones_like(indices, dtype=bool)
        adj_mat = sparse.csc_matrix((data, indices, indptr), shape=(len(gids), len(gids)))
        return cls(adj_mat, gids)

    def submatrix(self, sub_gids, sub_gids_post=None):
        """
        Return a submatrix specified by `sub_gids`
        :param sub_gids: Subpopulation to get the submatrix for. Can be either a list of gids, or an Assembly object
        :param sub_gids_post: (optiona) if specified, defines the postsynaptic population. Else pre- equals postsynaptic
        population
        :return: the adjacency submatrix of the specified population(s).
        """
        if sub_gids_post is not None:
            return self._m[np.ix_(self._lookup[self.__extract_gids__(sub_gids)],
                                     self._lookup[self.__extract_gids__(sub_gids_post)])]
        idx = self._lookup[self.__extract_gids__(sub_gids)]
        return self._m[np.ix_(idx, idx)]

    def dense_submatrix(self, sub_gids, sub_gids_post=None):
        return self.submatrix(sub_gids, sub_gids_post=sub_gids_post).todense()

    def subarray(self, sub_gids, sub_gids_post=None):
        return np.array(self.dense_submatrix(sub_gids, sub_gids_post=sub_gids_post))

    def sample_matrix_depth_profile(self, blueconfig_path, ref_gids):
        """
        Return a submatrix with the same (in propability) depth profile as `ref_gids`
        :param blueconfig_path: path to BlueConfig/CircuitConfig
        :param ref_gids: Subpopulation to use as reference for sampling.
                         Can be either a list of gids, or an Assembly object
        """
        from assemblyfire.utils import map_gids_to_depth

        ref_gids = self.__extract_gids__(ref_gids)
        assert np.isin(ref_gids, self._gids).all(), "Reference gids are not part of the connectivity matrix"

        # get depth values using bluepy
        depths_dict = map_gids_to_depth(blueconfig_path, self._gids)
        depths = np.array([depths_dict[gid] for gid in self._gids])
        ref_depths = depths[np.searchsorted(self._gids, ref_gids)]

        # bin reference and define probability for sampling
        hist, bin_edges = np.histogram(ref_depths, bins=50)
        p_ref = hist/np.sum(hist)
        depths_bins = np.digitize(depths, bins=bin_edges)
        assert len(hist == len(depths_bins[1:-1]))
        idx, counts = np.unique(depths_bins, return_counts=True)
        p = np.zeros_like(self._gids, dtype=np.float)
        for i in idx[1:-1]:
            p[depths_bins == i] = p_ref[i-1]/counts[i]
        if np.sum(p) != 1.0:  # this is a bit hacky but it solves small numerical errors
            # p /= np.sum(p)  # this doesn't work...
            idx = np.where(p != 0.)[0]
            p[idx[0]] += 1 - np.sum(p)
        assert(np.sum(p) == 1.0), "Probability for random sampling != 1.0"

        sample_gids = np.random.choice(self._gids, len(ref_gids), replace=False, p=p)
        idx = self._lookup[sample_gids]
        return self._m[np.ix_(idx, idx)]

    def dense_sample_depth_profile(self, blueconfig_path, ref_gids):
        return self.sample_matrix_depth_profile(blueconfig_path, ref_gids).todense()

    def sample_depth_profile(self, blueconfig_path, ref_gids):
        return np.array(self.dense_sample_depth_profile(blueconfig_path, ref_gids))

    def sample_matrix_mtype_composition(self, blueconfig_path, ref_gids):
        """
        Return a submatrix with the same (in propability) mtype composition as `ref_gids`
        :param blueconfig_path: path to BlueConfig/CircuitConfig
        :param ref_gids: Subpopulation to use as reference for sampling.
                         Can be either a list of gids, or an Assembly object
        """
        from assemblyfire.spikes import get_bluepy_simulation
        from assemblyfire.utils import get_mtypes

        ref_gids = self.__extract_gids__(ref_gids)
        assert np.isin(ref_gids, self._gids).all(), "Reference gids are not part of the connectivity matrix"

        # get mtypes using bluepy
        sim = get_bluepy_simulation(blueconfig_path)
        mtypes = np.asarray(get_mtypes(sim.circuit, self._gids))
        ref_mtypes = mtypes[np.searchsorted(self._gids, ref_gids)]

        # define probability for sampling based on reference
        mtypes_lst, counts = np.unique(ref_mtypes, return_counts=True)
        p_ref = counts / np.sum(counts)
        p = np.zeros_like(self._gids, dtype=np.float)
        for i, mtype in enumerate(mtypes_lst):
            idx = np.where(mtypes == mtype)[0]
            p[idx] = p_ref[i]/idx.shape[0]
        if np.sum(p) != 1.0:  # this is a bit hacky but it solves small numerical errors
            # p /= np.sum(p)  # this doesn't work...
            idx = np.where(p != 0.)[0]
            p[idx[0]] += 1 - np.sum(p)
        assert (np.sum(p) == 1.0), "Probability for random sampling != 1.0"

        sample_gids = np.random.choice(self._gids, len(ref_gids), replace=False, p=p)
        idx = self._lookup[sample_gids]
        return self._m[np.ix_(idx, idx)]

    def dense_sample_mtype_composition(self, blueconfig_path, ref_gids):
        return self.sample_matrix_mtype_composition(blueconfig_path, ref_gids).todense()

    def sample_mtype_composition(self, blueconfig_path, ref_gids):
        return np.array(self.dense_sample_mtype_composition(blueconfig_path, ref_gids))

    @classmethod
    def from_h5(cls, fn, group_name=None, prefix=None):
        from scipy import sparse
        if prefix is None:
            prefix = "connectivity"
        if group_name is None:
            prefix = "full_matrix"
        with h5py.File(fn, 'r') as h5:
            prefix_grp = h5[prefix]
            data_grp = prefix_grp[group_name]
            data = data_grp["data"][:]
            indices = data_grp["indices"][:]
            indptr = data_grp["indptr"][:]
            gids = data_grp["gids"][:]
            adj_mat = sparse.csc_matrix((data, indices, indptr), shape=(len(gids), len(gids)))
            return cls(adj_mat, gids)

    def to_h5(self, fn, group_name=None, prefix=None):
        if prefix is None:
            prefix = "connectivity"
        if group_name is None:
            prefix = "full_matrix"
        with h5py.File(fn, "a") as h5:
            prefix_grp = h5.require_group(prefix)
            data_grp = prefix_grp.create_group(group_name)
            data_grp.create_dataset("data", data=self._m.data)
            data_grp.create_dataset("indices", data=self._m.indices)
            data_grp.create_dataset("indptr", data=self._m.indptr)
            data_grp.create_dataset("gids", data=self._gids)


""" sample code for setup
import numpy as np
from assemblyfire.spikes import SpikeMatrixGroup, get_bluepy_simulation
from assemblyfire.connectivity import ConnectivityMatrix
from assemblyfire.utils import map_gids_to_depth, get_mtypes
spikes = SpikeMatrixGroup("/gpfs/bbp.cscs.ch/project/proj96/home/ecker/assemblyfire/configs/100p_depol_simmat.yaml")
conn_mat = ConnectivityMatrix.from_h5(spikes.h5f_name, group_name="full_matrix", prefix="connectivity")
blueconfig_path = spikes.get_blueconfig_path(spikes.seeds[0])
"""