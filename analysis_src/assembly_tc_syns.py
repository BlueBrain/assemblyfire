"""
Gets synapse (not connection) properties of projection synapses on assemblies
last modified: András Ecker 01.2023
"""

import os
from tqdm import tqdm
import numpy as np
import pandas as pd

from assemblyfire.config import Config
import assemblyfire.utils as utils
from assemblyfire.plots import plot_tc_syn_properties


def _get_spiking_proj_gids(config, sim_config, circuit_config):
    """Loads grouped (to patterns + non-specific) TC gids (that spike at least once)
    (Almost the same as in `assembly_topology.py`)"""
    proj_edge_pops = utils.get_proj_edge_pops(circuit_config, config.edge_pop)
    assert len(proj_edge_pops) <= 2, "The code assumes max 2 projections, one pattern specific and one non-specific"
    patterns_edge_pop = list(config.patterns_edges.values())[0]
    ns_edge_pop = np.setdiff1d(proj_edge_pops, [patterns_edge_pop])[0]
    pattern_gids = utils.get_pattern_node_idx(config.pattern_nodes_fname)

    proj_spikes = utils.get_proj_spikes(sim_config, config.t_start, config.t_end)
    proj_gids = {}
    for node_pop, spikes in proj_spikes.items():
        if node_pop == list(config.patterns_edges.keys())[0]:
            spiking_gids = spikes["spiking_gids"]
            for pattern_name, gids in pattern_gids.items():
                proj_gids[pattern_name] = {"edge_pop": patterns_edge_pop}
                proj_gids[pattern_name]["gids"] = np.unique(spiking_gids[np.in1d(spiking_gids, gids)])
        else:
            # ns_edge_pop[:3] gives "POm" which is just a bit less arbitrary than hard coding it...
            proj_gids[ns_edge_pop[:3]] = {"edge_pop": ns_edge_pop}
            proj_gids[ns_edge_pop[:3]]["gids"] = np.unique(spikes["spiking_gids"])
    return proj_gids


def get_tc2assembly_syn_properties(config, sim, assembly):
    """Finds synapse idx of patterns (+ non-specific stim) to assembly and loads synapse properties
    (loaded properties are hard coded atm. and require bluepy enums, but by this time bluepy should be installed)"""
    c = sim.circuit
    proj_gids = _get_spiking_proj_gids(config, sim.config, c.config)
    dfs = []
    for name, tmp in tqdm(proj_gids.items(), desc="Iterating over patterns"):
        syn_idx = utils.get_syn_idx(utils.get_edgef_name(c, tmp["edge_pop"]), tmp["gids"], assembly.gids)
        # print(len(syn_idx))
        # TODO: get rid of Synapse.POST_NEURITE_DISTANCE
        syn_df = utils.get_edge_properties(c, tmp["edge_pop"], syn_idx,
                                           ["@target_node", "conductance", Synapse.POST_NEURITE_DISTANCE])
        nrn_df = utils.get_node_properties(c, config.node_pop, syn_df["@target_node"].to_numpy(), ["layer", "mtype"])
        data = np.concatenate((syn_df[["conductance", Synapse.POST_NEURITE_DISTANCE]].to_numpy(),
                               nrn_df.loc[syn_df["@target_node"]].to_numpy()), axis=1)
        df = pd.DataFrame(data=data, index=syn_df.index.to_numpy(), columns=["g_syn", "path distance", "layer", "mtype"])
        df["name"] = name
        dfs.append(df.astype(dtype={"g_syn": np.float32, "path distance": np.float32, "layer": int}))
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

