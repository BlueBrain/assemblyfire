import os
from tqdm import tqdm
import numpy
import pandas

from conntility import subcellular
from conntility.connectivity import ConnectivityMatrix
import morphio
from assemblyfire.config import Config
from assemblyfire.dendritic_clustering_strength import DendriticClusteringResults as ClusteringResults
from assemblyfire.dendritic_clustering_strength import calculate_dendritic_clustering_strength as synapse_clustering_from_assemblies


str_assembly = "assembly{0}"

def innervation_from_assemblies(assembly_grp, conmat, tgt_gids):
    assembly_nconns = []
    for asmbly in assembly_grp.assemblies:
        asmbly_ncon = numpy.array(conmat.submatrix(asmbly.gids, sub_gids_post=tgt_gids).sum(axis=0))[0]
        assembly_nconns.append(asmbly_ncon)
    return numpy.vstack(assembly_nconns).transpose()

def innervation_clustering(circ, conmat, assembly_grp, fn_out, dset_root=None):
    from bluepy.connectome import Synapse  # TODO: SNAP instead
    numpy.random.seed(2345)

    exc_gids = circ.cells.ids("Excitatory")
    morph_root = os.path.join(os.path.split(circ.config["morphologies"])[0], "h5")
    morphs = circ.cells.get(conmat.gids, properties="morphology")

    buf_sz = 25
    gids_to_run = numpy.intersect1d(exc_gids, conmat.gids)
    L = len(gids_to_run)
    obj = ClusteringResults(fn_out, len(assembly_grp), dset_root=dset_root)
    gids_done = obj._df[("gid", "gid")].values.astype(int)
    assert numpy.all(numpy.in1d(gids_done, gids_to_run))
    gids_to_run = numpy.setdiff1d(gids_to_run, gids_done)
    gids_rnd = numpy.random.permutation(gids_to_run)

    indegree_mat = innervation_from_assemblies(assembly_grp, conmat, gids_rnd)

    pbar = tqdm(total=L, initial=obj._written)
    buf = []
    for gid, indegs in zip(gids_rnd, indegree_mat):
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
    import sys
    import bluepy
    import assemblyfire.utils as utils

    asmbly_grp = "seed1"
    fn_out = sys.argv[1]

    config = Config("../configs/v7_bbp-workflow.yaml")
    conmat = ConnectivityMatrix.from_h5(config.h5f_name, prefix=config.h5_prefix_connectivity,
                                    group_name="full_matrix")
    sim = bluepy.Simulation(os.path.join(config.root_path, "BlueConfig"))
    circ = sim.circuit
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)
    assembly_grp = assembly_grp_dict[asmbly_grp]

    tgt_gids = numpy.intersect1d(conmat.gids, circ.cells.ids("Excitatory"))

    innervation_clustering(circ, conmat, assembly_grp, fn_out, dset_root=asmbly_grp)
