"""
Class to handle synapse nearest neighbour distance results (and storage)
last modified: Michael Reimann 01.2023
"""

import h5py
import numpy as np
import pandas as pd


class SynNNDResults(object):
    """
    An object to hold results of synapse Nearest Neighbour Distance (NND) and read/write them from/to an HDF5 file.
    Currently, data access is through `obj._df`, a `pd.DataFrame`. More convenient access in the future.
    _df is a DataFrame where each row is a neuron, each column a type of result:
      - ("gid", "gid"): The gid of the neuron
      - ("assembly{0}", "strength"): The amount of clustering of synapses from assembly{0} onto the neuron.
                                     In terms of the z-score of the mean next-neighbor distance.
                                     Negative values indicate that dendritic locations are closer together than expected.
      - ("assembly{0}", "pvalue): The pvalue of a two-tailed ttest against the null hypothesis that
                                  dendritic locations associated with assembly{0} are randomly selected
                                  from all locations provided.
      - ("assembly{0}", "degree"): The in-degree from assembly{0} to the neuron
      - ("assembly{0}", "member"): 1 if the neuron is member of assembly{0}. 0 otherwise. 
    """
    DSET_PREFIX_DEFAULT = "syn_nnd"
    DSET_MEMBER = "member"
    DSET_CLST = "strength"
    DSET_PVALUE = "pvalue"
    DSET_DEG = "degree"

    def __init__(self, fn_out, n_assemblies, prefix=None):
        """
        param fn_out (str): Path to an HDF5 file to store the results in (or read them from).
        param n_assemblies (int): Number of assemblies that results will be calculated for.
        param prefix (str): Path within the HDF5 file to the results.
        return: Initializes the object. If the specified HDF5 file and the path inside exist, will
                try to read and initialize with the data contained inside; else the file will be created.
        """
        self._fn = fn_out
        self._written = 0
        if prefix is None:
            prefix = self.DSET_PREFIX_DEFAULT
        self.DSET_PREFIX = prefix
        
        df_dict = {("gid", "gid"): []}
        for i in range(n_assemblies):
            df_dict[("assembly%i" % i, self.DSET_MEMBER)] = []
            df_dict[("assembly%i" % i, self.DSET_CLST)] = []
            df_dict[("assembly%i" % i, self.DSET_PVALUE)] = []
            df_dict[("assembly%i" % i, self.DSET_DEG)] = []
        
        self._df = pd.DataFrame(df_dict)
        self._df = self._df[["gid"] + sorted([_x for _x in self._df.columns.levels[0] if _x != "gid"])]

        self._initialize_file(n_assemblies)
    
    @staticmethod
    def _sorted_df(df_in):
        tgt_order = ["gid"] + sorted([_x for _x in df_in.columns if _x != "gid"])
        return df_in[tgt_order]

    @staticmethod
    def _append_single(df, dset, i):
        assert dset.shape[0] == i
        o = len(df)
        if o == 0: return

        dset.resize((dset.shape[0] + o, dset.shape[1]))
        dset[-o:] = df
    
    def unwritten_rows(self):
        write_order = [self.DSET_MEMBER, self.DSET_CLST, self.DSET_PVALUE, self.DSET_DEG]
        unwritten = {}

        for str_dset in write_order:
            df = self._df.reorder_levels([1, 0], axis="columns")[["gid", str_dset]].droplevel(0, "columns")
            df = SynNNDResults._sorted_df(df)
            assert len(df) >= self._written
            unwritten[str_dset] = df.iloc[self._written:].values.astype(float)
        return unwritten
    
    def flush(self):
        """Write the clustering data to the underlying HDF5 file.
        (Only writes new data that is not already in the file.)"""
        unwritten = self.unwritten_rows()

        with h5py.File(self._fn, "a") as h5:
            for str_dset, df in unwritten.items():
                dset = h5[self.DSET_PREFIX][str_dset]
                self._append_single(df, dset, self._written)
            h5.flush()
        self._written = len(self._df)
    
    def append(self, other):
        """Append new results to the object.
        param other (pd.DataFrame): A pandas DataFrame with new results to append.
                                    Columns must be a subset of the columns in obj._df"""
        # TODO: Basic compatibility checks
        self._df = pd.concat([self._df, other], axis=0)

    def _initialize_file(self, n_assemblies):
        # TODO: URGENT: Save / restore column names in file, instead of this implicit encoding!
        h5 = h5py.File(self._fn, "a")

        grp = h5.require_group(self.DSET_PREFIX)
        existing_dict = {}
        for dset_str in [self.DSET_CLST, self.DSET_PVALUE, self.DSET_DEG, self.DSET_MEMBER]:
            if dset_str not in grp.keys():
                grp.create_dataset(dset_str, (0, n_assemblies + 1), float, maxshape=(None, n_assemblies + 1))
            else:
                dset = grp[dset_str]
                assert dset.shape[1] == (n_assemblies + 1)
                if ("gid", "gid") in existing_dict:
                    assert np.all(existing_dict[("gid", "gid")] == dset[:, 0].astype(int))
                else:
                    existing_dict[("gid", "gid")] = dset[:, 0].astype(int)
                assembly_names = sorted(["assembly%i" % i for i in range(n_assemblies)])
                for i, assembly_name in zip(range(n_assemblies), assembly_names):
                    existing_dict[(assembly_name, dset_str)] = dset[:, i + 1]
        self.append(pd.DataFrame.from_records(existing_dict))
        self._written = len(self._df)

