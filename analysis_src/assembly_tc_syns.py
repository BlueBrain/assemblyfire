"""
Gets synapse (not connection) properties of projection synapses on assemblies
last modified: András Ecker 06.2023
"""

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from morphio import Morphology
from conntility.subcellular import MorphologyPathDistanceCalculator

from assemblyfire.config import Config
import assemblyfire.utils as utils
from assemblyfire.plots import plot_tc_syn_properties

from assembly_topology import get_spiking_proj_gids


def get_tc2assembly_syn_properties(config, sim, assembly):
    """Finds synapse idx of patterns (+ non-specific stim) to assembly and loads synapse properties
    (loaded properties are hard coded atm. and require bluepy enums, but by this time bluepy should be installed)"""
    c = sim.circuit
    proj_gids, pattern_gids = get_spiking_proj_gids(config, sim.config, c.config)
    morph_root = c.config["components"]["morphologies_dir"]
    morphs = c.nodes[config.node_pop].get(assembly.gids, "morphology")
    soma_loc = pd.DataFrame({"afferent_section_id": [0], "afferent_segment_id": [0], "afferent_segment_offset": [0.0]})

    dfs = []
    for edge_pop, gids in proj_gids.items():
        syn_idx = utils.get_syn_idx(utils.get_edgef_name(c, edge_pop), gids, assembly.gids)
        syn_df = utils.get_edge_properties(c, edge_pop, syn_idx, ["@source_node", "@target_node", "conductance",
                                                                  "afferent_section_id", "afferent_segment_id",
                                                                  "afferent_segment_offset"])
        idx, path_distances = [], []
        for gid in tqdm(syn_df["@target_node"].unique(), desc="iterating over morphologies"):
            mpdc = MorphologyPathDistanceCalculator(Morphology(os.path.join(morph_root, morphs.loc[gid]) + ".asc"))
            syn_loc_df = syn_df.loc[syn_df["@target_node"] == gid, ["afferent_section_id", "afferent_segment_id",
                                                                    "afferent_segment_offset"]]
            idx.append(syn_loc_df.index.to_numpy())
            path_distances.append(mpdc.path_distances(syn_loc_df, soma_loc))
        pd_df = pd.DataFrame(data=np.vstack(path_distances), index=np.hstack(idx), columns=["path distance"])
        syn_df = syn_df.merge(pd_df, left_index=True, right_index=True)
        nrn_df = utils.get_node_properties(c, config.node_pop, syn_df["@target_node"].to_numpy(), ["layer", "mtype"])
        data = np.concatenate((syn_df[["@source_node", "conductance", "path distance"]].to_numpy(),
                               nrn_df[["layer", "mtype"]].to_numpy()), axis=1)
        df = pd.DataFrame(data=data, index=syn_df.index.to_numpy(), columns=["pre_gid", "g_syn", "path distance",
                                                                             "layer", "mtype"])
        df = df.astype(dtype={"pre_gid": int, "g_syn": np.float32, "path distance": np.float32, "layer": int})
        if edge_pop == list(config.patterns_edges.values())[0]:
            for pattern_name, gids_ in pattern_gids.items():
                pattern_df = df.loc[df["pre_gid"].isin(gids_)]
                pattern_df["name"] = pattern_name
                dfs.append(pattern_df)
        else:
            df["name"] = edge_pop[:3]  # gives "POm" which is just a bit less arbitrary than hard coding it...
            dfs.append(df)
    return pd.concat(dfs)


def main(config_path, seed, assembly_id):
    config = Config(config_path)
    sim = utils.get_bluepy_simulation(utils.get_sim_path(config.root_path).loc[seed])
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)
    assembly = assembly_grp_dict["seed%i" % seed].loc((assembly_id, seed))

    df = get_tc2assembly_syn_properties(config, sim, assembly)
    df.to_pickle("assembly%i_tc_syn_properties_seed%i.pkl" % (assembly_id, seed))
    fig_name = os.path.join(config.fig_path, "tc2assembly%i_syn_properties_seed%i.png" % (assembly_id, seed))
    plot_tc_syn_properties(df, fig_name)


if __name__ == "__main__":
    config_path = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/assemblyfire/configs/v7_10seeds_np.yaml"
    seed, assembly_id = 19, 8
    main(config_path, seed, assembly_id)

