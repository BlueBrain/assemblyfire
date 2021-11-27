# -*- coding: utf-8 -*-
"""
Class to get, save and load connection matrix and sample submatrices from it
authors: Michael Reimann, András Ecker
last modified: 11.2021
"""

import h5py
from tqdm import tqdm
from tqdm.contrib import tzip
import numpy as np
import pandas as pd
from scipy import sparse


REPORT_TO_SONATA = {"gmax_AMPA": "conductance", "rho": "rho0_GB"}


class _MatrixNodeIndexer(object):
    def __init__(self, parent, prop_name):
        self._parent = parent
        self._prop = parent._vertex_properties[prop_name]

    def eq(self, other):
        pop = self._parent._vertex_properties.index.values[self._prop == other]
        return self._parent.subpopulation(pop)

    def isin(self, other):
        pop = self._parent._vertex_properties.index.values[np.in1d(self._prop, other)]
        return self._parent.subpopulation(pop)

    def le(self, other):
        pop = self._parent._vertex_properties.index.values[self._prop <= other]
        return self._parent.subpopulation(pop)

    def lt(self, other):
        pop = self._parent._vertex_properties.index.values[self._prop < other]
        return self._parent.subpopulation(pop)

    def ge(self, other):
        pop = self._parent._vertex_properties.index.values[self._prop >= other]
        return self._parent.subpopulation(pop)

    def gt(self, other):
        pop = self._parent._vertex_properties.index.values[self._prop > other]
        return self._parent.subpopulation(pop)

    def random_numerical_gids(self, ref, n_bins=50):
        all_gids = self._prop.index.values
        ref_gids = self._parent.__extract_vertex_ids__(ref)
        assert np.isin(ref_gids, all_gids).all(), "Reference gids are not part of the connectivity matrix"

        ref_values = self._prop[ref_gids]
        hist, bin_edges = np.histogram(ref_values.values, bins=n_bins)
        bin_edges[-1] += (bin_edges[-1] - bin_edges[-2]) / 1E9
        value_bins = np.digitize(self._prop.values, bins=bin_edges)
        assert len(hist == len(value_bins[1:-1]))  # `digitize` returns values below and above the spec. bin_edges
        sample_gids = []
        for i in range(n_bins):
            idx = np.where(value_bins == i + 1)[0]
            assert idx.shape[0] >= hist[i], "Not enough neurons at this depths to sample from"
            sample_gids.extend(np.random.choice(all_gids[idx], hist[i], replace=False).tolist())
        return sample_gids

    def random_numerical(self, ref, n_bins=50):
        return self._parent.subpopulation(self.random_numerical_gids(ref, n_bins))

    def random_categorical_gids(self, ref):
        all_gids = self._prop.index.values
        ref_gids = self._parent.__extract_vertex_ids__(ref)
        assert np.isin(ref_gids, all_gids).all(), "Reference gids are not part of the connectivity matrix"

        ref_values = self._prop[ref_gids].values
        value_lst, counts = np.unique(ref_values, return_counts=True)
        sample_gids = []
        for i, value in enumerate(value_lst):
            idx = np.where(self._prop == value)[0]
            assert idx.shape[0] >= counts[i], "Not enough %s to sample from" % value
            sample_gids.extend(np.random.choice(all_gids[idx], counts[i], replace=False).tolist())
        return sample_gids

    def random_categorical(self, ref):
        return self._parent.subpopulation(self.random_categorical_gids(ref))


class _MatrixEdgeIndexer(object):
    def __init__(self, parent, prop_name):
        self._parent = parent
        self._prop = parent._edges[prop_name]

    def eq(self, other):
        idxx = self._prop == other
        return self._parent.subedges(idxx)

    def isin(self, other):
        idxx = np.isin(self._prop, other)
        return self._parent.subedges(idxx)

    def le(self, other):
        idxx = self._prop <= other
        return self._parent.subedges(idxx)

    def lt(self, other):
        idxx = self._prop < other
        return self._parent.subedges(idxx)

    def ge(self, other):
        idxx = self._prop >= other
        return self._parent.subedges(idxx)

    def gt(self, other):
        idxx = self._prop > other
        return self._parent.subedges(idxx)

    def full_sweep(self, direction='decreasing'):
        #  For an actual filtration. Take all values and sweep
        raise NotImplementedError()


def read_h5(fn, group_name="full_matrix", prefix="connectivity"):
    full_prefix = prefix + "/" + group_name
    edges = pd.read_hdf(fn, full_prefix + "/edges")
    vertex_properties = pd.read_hdf(fn, full_prefix + "/vertex_properties")

    with h5py.File(fn, 'r') as h5:
        data_grp = h5[full_prefix]
        shape = tuple(data_grp.attrs["NEUROTOP_SHAPE"])
        default_edge_property = data_grp.attrs["NEUROTOP_DEFAULT_EDGE"]
    return edges, vertex_properties, shape, default_edge_property


def il_groupby(df, group="local_syn_idx", by="pre_gid"):
    """Fast, pure numpy, in line implementation of `df.groupby(by)[group].apply(list)` (assumes that `by` is sorted)"""
    groupby = df[[group, by]].to_numpy()
    un_by, un_idx = np.unique(groupby[:, 1], return_index=True)
    return un_by, np.split(groupby[:, 0], un_idx[1:])


def from_report(sim, report_cfg, aggregation_fns=["sum", "mean"], precalculated={}, chunksize=1e4):
    """Creates weighted connectivity matrix from synapse report (written by Neurodamus, read by libsonata)
    TODO: parallelize the processing of chunks if memory permits..."""

    report = sim.report(report_cfg["report_name"])
    report_gids = report.gids
    # get exact time points from report (and the column indices for local to global synapse id mapping)
    data = report.get_gid(report_gids[0], t_start=report_cfg["t_start"],
                          t_end=report_cfg["t_end"], t_step=report_cfg["t_step"])

    if "gids" in precalculated and "edges" in precalculated:
        gids = precalculated["gids"]
        edges = precalculated["edges"]
    else:
        NotImplementedError()
        # TODO: write this properly for conntility
        # conn_mat = from_bluepy(sim)  # not sure of the syntax here...
        # edges = conn_mat._edges
        # gids = conn_mat._vertex_properties.index.to_numpy()
    if "syn_df" in precalculated:
        syn_df = precalculated["syn_df"]
    else:
        NotImplementedError()
        # TODO: write this properly for conntility
        # from map_syn_idx import create_syn_idx_df
        # syn_df = create_syn_idx_df(sim.circuit.config["connectome"], data.columns, pklf_name=None)
    if not np.isin(gids, report_gids, assume_unique=True).all():
        # these will be used later for getting the initial (and hopefully non-evolving) values with bluepy
        c = sim.circuit
        from bluepy.enums import Synapse
        sonata_rep_name = REPORT_TO_SONATA[report_cfg["report_name"]]

    # initialize array for storing the connectome at different time points
    report_t = data.index.to_numpy()
    extra_columns = ["%s_t=%i" % (fn, t) for fn in aggregation_fns for t in report_t]
    weighted_edges = np.zeros((edges.shape[0], len(extra_columns)), dtype=np.float32)

    i = 0
    n_ts = len(report_t)
    post_matrix_idx = edges["col"].unique()
    idx = np.arange(0, len(post_matrix_idx), chunksize, dtype=int)
    idx = np.append(idx, len(post_matrix_idx))
    for start_id, end_id in tzip(idx[:-1], idx[1:], desc="Loading chunks of gids from report"):
        post_matrix_idx_chunk = post_matrix_idx[start_id:end_id]
        post_gids = gids[post_matrix_idx_chunk]
        post_chunk_df = syn_df.loc[syn_df["post_gid"].isin(post_gids)]
        post_rep_gids = post_gids[np.isin(post_gids, report_gids, assume_unique=True)]
        data = report.get(gids=post_rep_gids, t_start=report_cfg["t_start"],
                          t_end=report_cfg["t_end"], t_step=report_cfg["t_step"])
        for post_matrix_id, post_gid in tzip(post_matrix_idx_chunk, post_gids, desc="Iterating over (post) gids",
                                             miniters=chunksize/100, leave=False):
            if post_gid in report_gids:
                post_df = post_chunk_df.loc[post_chunk_df["post_gid"] == post_gid]
                pre_gids, local_syn_idx = il_groupby(post_df)
                # assert (pre_gids.astype(int) == gids[edges[edges["col"] == post_matrix_id]["row"].values]).all()
                post_data = data[post_gid]  # first level indexing of report (post gid)
                for syn_idx in local_syn_idx:
                    conn_data = post_data[syn_idx]  # second level indexing of report (local syn idx)
                    for j, fn in enumerate(aggregation_fns):
                        weighted_edges[i, j*n_ts:(j+1)*n_ts] = getattr(conn_data, fn)(axis=1)
                    i += 1
            else:
                pre_gids = gids[edges[edges["col"] == post_matrix_id]["row"].values]
                nonrep_syn_df = c.connectome.pathway_synapses(pre_gids, post_gid, [sonata_rep_name, Synapse.PRE_GID])
                pre_gids_bluepy, conns_data = il_groupby(nonrep_syn_df, group=sonata_rep_name, by=Synapse.PRE_GID)
                # assert (pre_gids_bluepy.astype(int) == pre_gids).all()
                for conn_data in conns_data:
                    for j, fn in enumerate(aggregation_fns):
                        weighted_edges[i, j*n_ts:(j+1)*n_ts] = getattr(conn_data, fn)()
                    i += 1
    weighted_edges = pd.concat([edges, pd.DataFrame(data=weighted_edges, columns=extra_columns)], axis=1)
    weighted_edges.to_hdf("/gpfs/bbp.cscs.ch/project/proj96/scratch/home/ecker/"
                          "simulations/4073e95f-abb1-4b86-8c38-13cf9f00ce0b/weighted_edges.h5")


class ConnectivityMatrix(object):
    """Small utility class to hold a connections matrix and generate submatrices"""
    def __init__(self, *args, vertex_labels=None, vertex_properties=None,
                 edge_properties=None, default_edge_property="data", shape=None):
        """Not too intuitive init - please see `from_bluepy()` below"""
        # Initialization 1: By adjacency matrix
        if isinstance(args[0], np.ndarray) or isinstance(args[0], sparse.spmatrix):
            m = args[0]
            assert m.ndim == 2
            m = sparse.coo_matrix(m)
            self._edges = pd.DataFrame({
                "row": m.row,
                "col": m.col,
                "data": m.data
            })
            if shape is None:
                shape = m.shape
        # Initialization 2: By edge-specific DataFrames
        if isinstance(args[0], pd.DataFrame):
            assert "row" in args[0] and "col" in args[0]
            self._edges = args[0]
            if shape is None:
                shape = (np.max(self._edges["row"]), np.max(self._edges["col"]))

        # In the future: implement the ability to represent connectivity from population A to B.
        # For now only connectivity within one and the same population
        assert shape[0] == shape[1]
        self._shape = shape

        # Initialize vertex property DataFrame
        if vertex_properties is None:
            if vertex_labels is None:
                vertex_labels = np.arange(shape[0])
            self._vertex_properties = pd.DataFrame({}, index=vertex_labels)
        elif isinstance(vertex_properties, dict):
            if vertex_labels is None:
                vertex_labels = np.arange(shape[0])
            self._vertex_properties = pd.DataFrame(vertex_properties, index=vertex_labels)
        elif isinstance(vertex_properties, pd.DataFrame):
            if vertex_labels is not None:
                raise ValueError("""Cannot specify vertex labels separately
                                 when instantiating vertex_properties explicitly""")
            self._vertex_properties = vertex_properties
        else:
            raise ValueError("""When specifying vertex properties it must be a DataFrame or dict""")
        assert len(self._vertex_properties) == shape[0]

        # Adding additional edge properties
        if edge_properties is not None:
            for prop_name, prop_mat in edge_properties.items():
                self.add_edge_property(prop_name, prop_mat)

        self._default_edge = default_edge_property

        self._lookup = self.__make_lookup__()
        #  NOTE: This part implements the .gids and .depth properties
        for colname in self._vertex_properties.columns:
            #  TODO: Check colname against existing properties
            setattr(self, colname, self._vertex_properties[colname].values)

        # TODO: calling it "gids" might be too BlueBrain-specific! Change name?
        self.gids = self._vertex_properties.index.values

    def __len__(self):
        return len(self.gids)

    def add_vertex_property(self, new_label, new_values):
        assert len(new_values) == len(self), "New values size mismatch"
        assert new_label not in self._vertex_properties, "Property {0} already exists!".format(new_label)
        self._vertex_properties[new_label] = new_values

    def add_edge_property(self, new_label, new_values):
        if (isinstance(new_values, np.ndarray) and new_values.ndim == 2) or isinstance(new_values, sparse.spmatrix):
            new_values = sparse.coo_matrix(new_values)
            assert np.all(new_values.row == self._edges["row"]) and np.all(new_values.col == self._edges["col"])
            self._edges[new_label] = new_values.data
        else:
            assert len(new_values) == len(self._edges)
            self._edges[new_label] = new_values

    def __make_lookup__(self):
        return pd.Series(np.arange(self._shape[0]), index=self._vertex_properties.index)

    @property
    def edge_properties(self):
        # TODO: Maybe exclude "row" and "col"?
        return self._edges.columns.values

    @property
    def vertex_properties(self):
        return self._vertex_properties.columns.values

    def matrix_(self, edge_property=None):
        if edge_property is None:
            edge_property = self._default_edge
        return sparse.coo_matrix((self._edges[edge_property], (self._edges["row"], self._edges["col"])),
                                 shape=self._shape)

    @property
    def matrix(self):
        return self.matrix_(self._default_edge)

    def dense_matrix_(self, edge_property=None):
        return self.matrix_(edge_property=edge_property).todense()

    @property
    def dense_matrix(self):
        return self.dense_matrix_()

    def array_(self, edge_property=None):
        return np.array(self.dense_matrix_(edge_property=edge_property))

    @property
    def array(self):
        return self.array_()

    def index(self, prop_name):
        assert prop_name in self._vertex_properties, "vertex property should be in " + str(self.vertex_properties)
        return _MatrixNodeIndexer(self, prop_name)

    def filter(self, prop_name=None):
        if prop_name is None:
            prop_name = self._default_edge
        return _MatrixEdgeIndexer(self, prop_name)

    def default(self, new_default_property):
        assert new_default_property in self.edge_properties, "Edge property {0} unknown!".format(new_default_property)
        return ConnectivityMatrix(self._edges, vertex_properties=self._vertex_properties, shape=self._shape,
                                  default_edge_property=new_default_property)

    @staticmethod
    def __extract_vertex_ids__(an_obj):
        from assemblyfire.assemblies import Assembly
        if isinstance(an_obj, Assembly):
            return an_obj.gids
        return an_obj

    @classmethod
    def from_bluepy(cls, blueconfig_path, gids=None):
        """
        BlueConfig based constructor
        :param blueconfig_path: path to BlueConfig
        :param gids: array of gids AKA. the nodes of the graph, if None - all excitatory gids from the circuit are used
        """
        from scipy import sparse
        from assemblyfire.spikes import get_bluepy_simulation
        from assemblyfire.utils import get_E_gids, get_mtypes, get_figure_asthetics

        sim = get_bluepy_simulation(blueconfig_path)
        if gids is None:
            gids = get_E_gids(sim.circuit, sim.target)
        depths, _ = get_figure_asthetics(blueconfig_path, sim.target, gids)
        mtypes = get_mtypes(sim.circuit, gids)
        conv = pd.Series(np.arange(len(gids)), index=gids)
        indptr = [0]
        indices = []
        for gid in tqdm(gids, desc="Building connectivity matrix", miniters=len(gids) / 100):
            aff = conv[np.intersect1d(sim.circuit.connectome.afferent_gids(gid), gids)]
            indices.extend(aff)
            indptr.append(len(indices))
        data = np.ones_like(indices, dtype=bool)
        adj_mat = sparse.csc_matrix((data, indices, indptr), shape=(len(gids), len(gids)))
        vertex_props = pd.DataFrame({"depths": depths.to_numpy().reshape(-1),
                                     "mtypes": mtypes.to_numpy().reshape(-1)},
                                    index=gids)
        return cls(adj_mat, vertex_properties=vertex_props)

    def submatrix(self, sub_gids, edge_property=None, sub_gids_post=None):
        """Return a submatrix specified by `sub_gids`"""
        m = self.matrix_(edge_property=edge_property).tocsc()
        if sub_gids_post is not None:
            return m[np.ix_(self._lookup[self.__extract_vertex_ids__(sub_gids)],
                            self._lookup[self.__extract_vertex_ids__(sub_gids_post)])]
        idx = self._lookup[self.__extract_vertex_ids__(sub_gids)]
        return m[np.ix_(idx, idx)]

    def dense_submatrix(self, sub_gids, edge_property=None, sub_gids_post=None):
        return self.submatrix(sub_gids, edge_property=edge_property, sub_gids_post=sub_gids_post).todense()

    def subarray(self, sub_gids, edge_property=None, sub_gids_post=None):
        return np.array(self.dense_submatrix(sub_gids, edge_property=edge_property, sub_gids_post=sub_gids_post))

    def subpopulation(self, subpop_ids, copy=True):
        """A ConnectivityMatrix object representing the specified subpopulation"""
        subpop_ids = self.__extract_vertex_ids__(subpop_ids)
        if not copy:
            #  TODO: Return a view on this object
            raise NotImplementedError()
        assert np.all(np.in1d(subpop_ids, self._vertex_properties.index.values))

        tmp_submat = self.submatrix(subpop_ids).tocoo()
        out_edges = {"row": tmp_submat.row, "col": tmp_submat.col}
        for edge_prop in self.edge_properties:
            if edge_prop not in ["row", "col"]:
                out_edges[edge_prop] = self.submatrix(subpop_ids, edge_property=edge_prop).data
        out_edges = pd.DataFrame(out_edges)
        out_vertices = self._vertex_properties.loc[subpop_ids]
        return ConnectivityMatrix(out_edges, vertex_properties=out_vertices, shape=(len(subpop_ids), len(subpop_ids)),
                                  default_edge_property=self._default_edge)

    def subedges(self, subedge_indices, copy=True):
        """A ConnectivityMatrix object representing the specified subpopulation"""
        if not copy:
            #  TODO: Return a view on this object
            raise NotImplementedError()

        if subedge_indices.dtype == bool:
            out_edges = self._edges[subedge_indices]
        else:
            out_edges = self._edges.iloc[subedge_indices]
        return ConnectivityMatrix(out_edges, vertex_properties=self._vertex_properties, shape=self._shape,
                                  default_edge_property=self._default_edge)

    def random_n_gids(self, ref):
        """Randomly samples `ref` number of neurons if `ref` is and int,
        otherwise the same number of neurons as in `ref`"""
        all_gids = self._vertex_properties.index.values
        if hasattr(ref, "__len__"):
            assert np.isin(self.__extract_vertex_ids__(ref),
                           all_gids).all(), "Reference gids are not part of the connectivity matrix"
            n_samples = len(ref)
        elif isinstance(ref, int):  # Just specify the number
            n_samples = ref
        else:
            raise ValueError("random_n_gids() has to be called with an int or something that has len()")
        return np.random.choice(all_gids, n_samples, replace=False)

    def random_n(self, ref):
        return self.subpopulation(self.random_n_gids(ref))

    @classmethod
    def from_h5(cls, fn, group_name="full_matrix", prefix="connectivity"):
        edges, vertex_properties, shape, default_edge_property = read_h5(fn, group_name, prefix)
        return cls(edges, vertex_properties=vertex_properties,
                   default_edge_property=default_edge_property, shape=shape)

    def to_h5(self, fn, group_name=None, prefix=None):
        if prefix is None:
            prefix = "connectivity"
        if group_name is None:
            group_name = "full_matrix"
        full_prefix = prefix + "/" + group_name
        self._vertex_properties.to_hdf(fn, key=full_prefix + "/vertex_properties")
        self._edges.to_hdf(fn, key=full_prefix + "/edges")

        with h5py.File(fn, "a") as h5:
            data_grp = h5[full_prefix]
            data_grp.attrs["NEUROTOP_SHAPE"] = self._shape
            data_grp.attrs["NEUROTOP_DEFAULT_EDGE"] = self._default_edge
            data_grp.attrs["NEUROTOP_CLASS"] = "ConnectivityMatrix"


if __name__ == "__main__":
    from bluepy import Simulation
    sim = Simulation("/gpfs/bbp.cscs.ch/project/proj96/scratch/home/ecker/"
                     "simulations/LayerWiseEShotNoise_PyramidPatterns/BlueConfig")
    edges, vertex_properties, _, _ = read_h5("/gpfs/bbp.cscs.ch/project/proj96/scratch/home/ecker/"
                                             "simulations/4073e95f-abb1-4b86-8c38-13cf9f00ce0b/assemblies.h5")
    syn_df = pd.read_pickle("/gpfs/bbp.cscs.ch/project/proj96/circuits/plastic_v1/syn_idx.pkl")
    precalculated = {"gids": vertex_properties.index.to_numpy(), "edges": edges, "syn_df": syn_df}
    report_cfg = {"report_name": "gmax_AMPA", "t_start": 1500, "t_end": 62000, "t_step": 2000}
    from_report(sim, report_cfg, precalculated=precalculated)

