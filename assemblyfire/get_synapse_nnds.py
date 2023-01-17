"""
Main run function for getting synapse nearest neighbour distances
authors: Michael Reimann, András Ecker; last modified: 01.2023
"""

import os
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
from morphio import Morphology
from conntility.subcellular import MorphologyPathDistanceCalculator

from assemblyfire.config import Config
import assemblyfire.utils as utils
from assemblyfire.topology import AssemblyTopology
from assemblyfire.syn_nnd import SynNNDResults
from assemblyfire.clustering import syn_nearest_neighbour_distances

L = logging.getLogger("assemblyfire")


def _assembly_group_from_name(config, assembly_grp_name):
    """Loads in the correct AssemblyGroup"""
    if assembly_grp_name == "consensus":
        cons_assemblies = utils.load_consensus_assemblies_from_h5(config.h5f_name, config.h5_prefix_consensus_assemblies)
        return utils.consensus_dict2assembly_grp(cons_assemblies)
    elif assembly_grp_name == "seed_average":
        assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_avg_assemblies)
        return assembly_grp_dict["seed_average"]
    else:
        assert "seed" in assembly_grp_name, "Need to specify a seed, `seed_average`, or `consensus`"
        assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)
        return assembly_grp_dict[assembly_grp_name]


def _get_assembly_indegrees(assembly_grp, conn_mat, gids):
    """Gets indegrees from each assembly in the `assembly_grp` for all `gids`"""
    assembly_indegrees = [conn_mat.degree(assembly.gids, gids) for assembly in assembly_grp.assemblies]
    return np.vstack(assembly_indegrees).transpose()


def run(config_path, assembly_grp_name, buf_size, seed):
    """TODO"""
    from bluepy.enums import Synapse

    config = Config(config_path)
    assembly_grp = _assembly_group_from_name(config, assembly_grp_name)
    conn_mat = AssemblyTopology.from_h5(config.h5f_name, prefix=config.h5_prefix_connectivity)
    c = utils.get_bluepy_circuit(utils.get_sim_path(config.root_path).iloc[0])
    gids = utils.get_gids(c, config.target)
    morph_root = os.path.join(os.path.split(c.config["morphologies"])[0], "h5")
    morphs = c.cells.get(gids, properties="morphology")
    results = SynNNDResults(config.h5f_name, len(assembly_grp), prefix="%s_syn_nnd" % assembly_grp_name)

    # check which gids are already done, and shuffle the (order of the) remaining ones
    gids2run = np.intersect1d(gids, conn_mat.gids)  # just to make sure...
    total = len(gids2run)
    gids_done = np.sort(np.unique(results._df[("gid", "gid")].values.astype(int)))
    assert np.in1d(gids_done, gids2run).all()
    gids2run = np.setdiff1d(gids2run, gids_done, assume_unique=True)
    np.random.seed(seed)
    gids_rnd = np.random.permutation(gids2run)
    L.info(" Getting synapse nearest neighbour distance for %i / %i gids " % (len(gids_rnd), len(gids)))

    indegree_mat = _get_assembly_indegrees(assembly_grp, conn_mat, gids_rnd)

    pbar, buf = tqdm(total=total, initial=results._written), []
    for gid, gid_indegrees in zip(gids_rnd, indegree_mat):
        pbar.update()
        # prepare args, and calculate synapse nearest neighbour distance
        mpdc = MorphologyPathDistanceCalculator(Morphology(os.path.join(morph_root, morphs.loc[gid]) + ".h5"))
        syn_loc_df = c.connectome.afferent_synapses(gid, properties=["afferent_section_id", "afferent_segment_id",
                                                                     "afferent_segment_offset", Synapse.PRE_GID
                                                                     ]).set_index(Synapse.PRE_GID)
        syn_loc_df = syn_loc_df.loc[syn_loc_df.index.intersection(gids)]
        clst_dict = syn_nearest_neighbour_distances(gid, mpdc, syn_loc_df, assembly_grp)
        # adding assembly indegrees to the dataset (for faster analysis afterwards)
        for assembly, assembly_indegree in zip(assembly_grp, gid_indegrees):
            clst_dict[("assembly%i" % assembly.idx[0], SynNNDResults.DSET_DEG)] = assembly_indegree
        # write to file when buffer is full
        buf.append(clst_dict)
        if len(buf) >= buf_size:
            results.append(pd.DataFrame.from_records(buf))
            results.flush()
            buf = []
    # write the last part of the results to file as well...
    results.append(pd.DataFrame.from_records(buf))
    results.flush()

