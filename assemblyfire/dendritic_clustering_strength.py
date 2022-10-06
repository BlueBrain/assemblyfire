import os
from tqdm import tqdm
import h5py
import numpy
import pandas

from scipy.stats import ttest_ind

str_assembly = "assembly{0}"


class DendriticClusteringResults(object):
    """
    An object to hold results of dendritic clustering strengths and read/write them from/to an hdf5 file.
    Currently, data access is through obj._df, a pandas.DataFrame. More convenient access in the future.

    _df is a DataFrame where each row is a neuron, each column a type of result:
      - ("gid", "gid"): The gid of the neuron
      - ("assembly{0}", "strength"): The amount of clustering of synapses from assembly{0} onto the neuron. In terms
      of the z-score of the mean next-neighbor distance. Negative values indicate that dendritic locations are
      closer together than expected.
      - ("assembly{0}", "pvalue): The pvalue of a two-tailed ttest against the null hypothesis that dendritic locations
      associated with assembly{0} are randomly selected from all locations provided.
      - ("assembly{0}", "degree"): The in-degree from assembly{0} to the neuron
      - ("assembly{0}", "member"): 1 if the neuron is member of assembly{0}. 0 otherwise. 
    """
    DSET_ROOT_DEFAULT = "clustering"
    DSET_MEMBER = "member"
    DSET_CLST = "strength"
    DSET_PVALUE = "pvalue"
    DSET_DEG = "degree"

    def __init__(self, fn_out, n_assemblies, dset_root=None):
        """
        Args:
          fn_out (str): Path to an hdf5 file to store the results in (or read them from).
          n_assemblies (int): Number of assemblies that results will be calculated for.
          dset_root (str): Path within the hdf5 file to the results.
        
        Returns:
          Initializes the object. If the specified hdf5 file and the path inside exist, will
          try to read and initialize with the data contained inside; else the file will be created.
        """
        self._fn = fn_out
        self._written = 0
        if dset_root is None:
            dset_root = DendriticClusteringResults.DSET_ROOT_DEFAULT
        self.DSET_ROOT = dset_root
        
        df_dict = {("gid", "gid"): []}
        for i in range(n_assemblies):
            df_dict[("assembly{0}".format(i), self.DSET_MEMBER)] = []
            df_dict[("assembly{0}".format(i), self.DSET_CLST)] = []
            df_dict[("assembly{0}".format(i), self.DSET_PVALUE)] = []
            df_dict[("assembly{0}".format(i), self.DSET_DEG)] = []
        
        self._df = pandas.DataFrame(df_dict)
        self._df = self._df[["gid"] + sorted([_x for _x in self._df.columns.levels[0] if _x != "gid"])]

        self._initialize_file(n_assemblies)
    
    @staticmethod
    def _sorted_df(df_in):
        tgt_order = ["gid"] + sorted([_x for _x in df_in.columns if _x != "gid"])
        return df_in[tgt_order]

    @staticmethod
    def _write_single(df, dset, i):
        df = DendriticClusteringResults._sorted_df(df)
        assert dset.shape[0] == i
        assert len(df) > i
        dset.resize((len(df), dset.shape[1]))
        o = len(df) - i
        dset[-o:] = df.iloc[i:].values
    
    def flush(self):
        """
        Write the clustering data to the underlying hdf5 file. Only writes new data that is not already in the file! 
        """
        with h5py.File(self._fn, "a") as h5:
            for str_dset in [self.DSET_MEMBER, self.DSET_CLST, self.DSET_PVALUE, self.DSET_DEG]:
                df = self._df.reorder_levels([1, 0], axis="columns")[["gid", str_dset]].droplevel(0, "columns")
                dset = h5[self.DSET_ROOT][str_dset]
                self._write_single(df, dset, self._written)
            h5.flush()
        self._written = len(self._df)
    
    def append(self, other):
        """
        Append new results to the object.
        Args:
          other (pandas.DataFrame): A pandas DataFrame with new results to append. Columns must be a subset of the columns
          in obj._df
        """
        # TODO: Basic compatibility checks
        self._df = pandas.concat([self._df, other], axis=0)

    def _initialize_file(self, n_assemblies):
        h5 = h5py.File(self._fn, "a")

        grp = h5.require_group(self.DSET_ROOT)
        existing_dict = {}
        for dset_str in [self.DSET_CLST, self.DSET_PVALUE, self.DSET_DEG, self.DSET_MEMBER]:
            if dset_str not in grp.keys():
                grp.create_dataset(dset_str, (0, n_assemblies + 1), float, maxshape=(None, n_assemblies + 1))
            else:
                dset = grp[dset_str]
                assert dset.shape[1] == (n_assemblies + 1)
                if ("gid", "gid") in existing_dict:
                    assert numpy.all(existing_dict[("gid", "gid")] == dset[:, 0])
                else:
                    existing_dict[("gid", "gid")] = dset[:, 0].astype(int)
                for i in range(n_assemblies):
                    existing_dict[("assembly{0}".format(i), dset_str)] = dset[:, i + 1]
        self.append(pandas.DataFrame.from_records(existing_dict))
        self._written = len(self._df)


def calculate_dendritic_clustering_strength(gid, mpdc, dendritic_locations, assemblies, consider_only_same_section=False):
    """
    Calculate the clustering strengths of dendritic locations.

    Args:
      gid (int): gid of the neuron whose dendritic locations are considered
      mpdc (conntility.subcellular.MorphologyPathDistanceCalculator): Path distance calculator for the same neuron
      dendritic_locations (pandas.DataFrame): frame specifying dendritic locations on the dendrite. Each row one location.
      Columns are: ["afferent_section_id", "afferent_segment_id", "afferent_segment_offset"]. Indexed by the (presynaptic)
      gids associated with the locations.
      assemblies (AssemblyGroup): group of assemblies that are to be tested for clustering. The test is whether the dendritic
      locations associated with the assembly.gids are closer together than the location associated with an equally sized
      group.
      consider_only_same_section (bool, default=False): If True, only locations on the same dendritic section are considered.
    """
    results = {}
    for assembly in assemblies:
        ugids = numpy.unique(dendritic_locations.index)
        from_assembly = numpy.in1d(dendritic_locations.index.values, assembly.gids)
        from_assembly_count = len(numpy.intersect1d(dendritic_locations.index.values, assembly.gids))
        if from_assembly.sum() == 0:
            results.update({
                ("gid", "gid"): gid,
                (str_assembly.format(assembly.idx[0]), DendriticClusteringResults.DSET_CLST): numpy.NaN,
                (str_assembly.format(assembly.idx[0]), DendriticClusteringResults.DSET_PVALUE): numpy.NaN,
                (str_assembly.format(assembly.idx[0]), DendriticClusteringResults.DSET_MEMBER): gid in assembly.gids
            })
            continue
        pd_data = mpdc.path_distances(dendritic_locations[from_assembly], same_section_only=consider_only_same_section)
        pd_data[pd_data == 0] = numpy.NaN
        nn_data = numpy.nanmin(pd_data, axis=0)

        nn_ctrl = []
        for seed in range(20):
            from_assembly_ = numpy.in1d(dendritic_locations.index.values, numpy.random.choice(ugids, from_assembly_count, replace=False))
            pd_ctrl = mpdc.path_distances(dendritic_locations[from_assembly_], same_section_only=consider_only_same_section)
            pd_ctrl[pd_ctrl == 0] = numpy.NaN
            nn_ctrl.append(numpy.nanmin(pd_ctrl, axis=0))
        a = numpy.mean(nn_data)
        b = [numpy.mean(_ctrl) for _ctrl in nn_ctrl]

        str_res = (a - numpy.nanmean(b)) / numpy.nanstd(b)
        stat_res = ttest_ind(nn_data, numpy.hstack(nn_ctrl), nan_policy='omit')
        mmbr_res = gid in assembly.gids

        results[("gid", "gid")] = gid
        results[(str_assembly.format(assembly.idx[0]), DendriticClusteringResults.DSET_CLST)] = str_res
        results[(str_assembly.format(assembly.idx[0]), DendriticClusteringResults.DSET_PVALUE)] = stat_res.pvalue
        results[(str_assembly.format(assembly.idx[0]), DendriticClusteringResults.DSET_MEMBER)] = mmbr_res
    return results
