import os
from tqdm import tqdm
import h5py
import numpy
import pandas

from conntility import subcellular
from conntility.connectivity import ConnectivityMatrix
from scipy.stats import ttest_ind
import morphio
from assemblyfire.config import Config


str_assembly = "assembly{0}"

class ClusteringResults(object):
    DSET_ROOT_DEFAULT = "clustering"
    DSET_MEMBER = "member"
    DSET_CLST = "strength"
    DSET_PVALUE = "pvalue"
    DSET_DEG = "degree"

    def __init__(self, fn_out, n_assemblies, dset_root=None):
        self._fn = fn_out
        self._written = 0
        if dset_root is None:
            dset_root = ClusteringResults.DSET_ROOT_DEFAULT
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
        df = ClusteringResults._sorted_df(df)
        assert dset.shape[0] == i
        assert len(df) > i
        dset.resize((len(df), dset.shape[1]))
        o = len(df) - i
        dset[-o:] = df.iloc[i:].values
    
    def flush(self):
        with h5py.File(self._fn, "a") as h5:
            for str_dset in [self.DSET_MEMBER, self.DSET_CLST, self.DSET_PVALUE, self.DSET_DEG]:
                df = self._df.reorder_levels([1, 0], axis="columns")[["gid", str_dset]].droplevel(0, "columns")
                dset = h5[self.DSET_ROOT][str_dset]
                self._write_single(df, dset, self._written)
        self._written = len(self._df)
    
    def append(self, other):
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


def synapse_clustering_from_assemblies(gid, mpdc, innervating_syns, assemblies, consider_only_same_section=False):

    results = {}
    for assembly in assemblies:
        ugids = numpy.unique(innervating_syns.index)
        from_assembly = numpy.in1d(innervating_syns.index.values, assembly.gids)
        from_assembly_count = len(numpy.intersect1d(innervating_syns.index.values, assembly.gids))
        if from_assembly.sum() == 0:
            results.update({
                ("gid", "gid"): gid,
                (str_assembly.format(assembly.idx[0]), ClusteringResults.DSET_CLST): numpy.NaN,
                (str_assembly.format(assembly.idx[0]), ClusteringResults.DSET_PVALUE): numpy.NaN,
                (str_assembly.format(assembly.idx[0]), ClusteringResults.DSET_MEMBER): gid in assembly.gids
            })
            continue
        pd_data = mpdc.path_distances(innervating_syns[from_assembly], same_section_only=consider_only_same_section)
        pd_data[pd_data == 0] = numpy.NaN
        nn_data = numpy.nanmin(pd_data, axis=0)

        nn_ctrl = []
        for seed in range(20):
            from_assembly_ = numpy.in1d(innervating_syns.index.values, numpy.random.choice(ugids, from_assembly_count, replace=False))
            pd_ctrl = mpdc.path_distances(innervating_syns[from_assembly_], same_section_only=consider_only_same_section)
            pd_ctrl[pd_ctrl == 0] = numpy.NaN
            nn_ctrl.append(numpy.nanmin(pd_ctrl, axis=0))
        a = numpy.mean(nn_data)
        b = [numpy.mean(_ctrl) for _ctrl in nn_ctrl]

        str_res = (a - numpy.nanmean(b)) / numpy.nanstd(b)
        stat_res = ttest_ind(nn_data, numpy.hstack(nn_ctrl), nan_policy='omit')
        mmbr_res = gid in assembly.gids

        results[("gid", "gid")] = gid
        results[(str_assembly.format(assembly.idx[0]), ClusteringResults.DSET_CLST)] = str_res
        results[(str_assembly.format(assembly.idx[0]), ClusteringResults.DSET_PVALUE)] = stat_res.pvalue
        results[(str_assembly.format(assembly.idx[0]), ClusteringResults.DSET_MEMBER)] = mmbr_res
    return results

def innervation_from_assemblies(assembly_grp, conmat, tgt_gids):
    assembly_nconns = []
    for asmbly in assembly_grp.assemblies:
        asmbly_ncon = numpy.array(conmat.submatrix(asmbly.gids, sub_gids_post=tgt_gids).sum(axis=0))[0]
        assembly_nconns.append(asmbly_ncon)
    return numpy.vstack(assembly_nconns).transpose()

def innervation_clustering(circ, conmat, assembly_grp, fn_out, dset_root=None):
    from bluepy.connectome import Synapse  # TODO: SNAP instead

    exc_gids = circ.cells.ids("Excitatory")
    morph_root = os.path.join(os.path.split(circ.config["morphologies"])[0], "h5")
    morphs = circ.cells.get(conmat.gids, properties="morphology")

    numpy.random.seed(2345)
    buf_sz = 25
    gids_rnd = numpy.random.permutation(numpy.intersect1d(exc_gids, conmat.gids))
    indegree_mat = innervation_from_assemblies(assembly_grp, conmat, gids_rnd)

    obj = ClusteringResults(fn_out, len(assembly_grp), dset_root=dset_root)
    offset = obj._written

    pbar = tqdm(total=len(gids_rnd), initial=offset)
    buf = []
    for gid, indegs in zip(gids_rnd[offset:], indegree_mat[offset:]):
        pbar.update()
        mm = morphio.Morphology(os.path.join(morph_root, morphs.loc[gid]) + ".h5")
        mpdc = subcellular.MorphologyPathDistanceCalculator(mm)
        innervating_syns = circ.connectome.afferent_synapses(gid, properties=[
            "afferent_section_id",
            "afferent_segment_id",
            "afferent_segment_offset",
            Synapse.PRE_GID
        ]).set_index(Synapse.PRE_GID)
        innervating_exc = innervating_syns.loc[innervating_syns.index.intersection(exc_gids)]

        clst_dict = synapse_clustering_from_assemblies(gid, mpdc, innervating_exc, assembly_grp)
        for asmbly, assembly_indeg in zip(assembly_grp, indegs):
            clst_dict[(str_assembly.format(asmbly.idx[0]), ClusteringResults.DSET_DEG)] = assembly_indeg

        buf.append(clst_dict)
        if len(buf) >= buf_sz:
            obj.append(pandas.DataFrame.from_records(buf))
            buf = []
            obj.flush()

if __name__ == "__main__":
    import bluepy
    import assemblyfire.utils as utils

    asmbly_grp = "seed1"

    config = Config("../configs/v7_bbp-workflow.yaml")
    conmat = ConnectivityMatrix.from_h5(config.h5f_name, prefix=config.h5_prefix_connectivity,
                                    group_name="full_matrix")
    sim = bluepy.Simulation(os.path.join(config.root_path, "BlueConfig"))
    circ = sim.circuit
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)
    assembly_grp = assembly_grp_dict[asmbly_grp]

    tgt_gids = numpy.intersect1d(conmat.gids, circ.cells.ids("Excitatory"))

    innervation_clustering(circ, conmat, assembly_grp, "clustering_tests.h5", dset_root=asmbly_grp)
