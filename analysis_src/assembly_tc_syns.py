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


def _get_tc_data(config, sim_config):
    """Loads grouped (to patterns + non-specific) TC spikes, and the edge file name associated with the projection
    (Could be done easier with adding stuff to the yaml config... but whatever)"""
    tc_spikes = utils.get_grouped_tc_spikes(config.pattern_gids_fname, sim_config, config.t_start, config.t_end)
    _, patterns = utils.get_stimulus_stream(config.input_patterns_fname, config.t_start, config.t_end)
    pattern_names = np.unique(patterns)
    projf_names = utils.get_projf_names(sim_config)
    assert len(projf_names) <= 2, "The code assumes max 2 projections, one pattern specific and one non-specific"
    ns_proj_name = np.setdiff1d(list(projf_names.keys()), [config.patterns_projection_name])[0]
    for name, data in tc_spikes.items():
        if name in pattern_names:
            data["proj_name"] = config.patterns_projection_name
            data["edgef_name"] = projf_names[config.patterns_projection_name]
        else:
            data["proj_name"] = ns_proj_name
            data["edgef_name"] = projf_names[ns_proj_name]
    return tc_spikes


def get_tc2assembly_syn_properties(config, sim, assembly):
    """Finds synapse idx of patterns (+ non-specific stim) to assembly and loads synapse properties
    (loaded properties are hard coded atm. and require bluepy enums, but by this time bluepy should be installed)"""
    from bluepy.enums import Cell, Synapse
    tc_spikes = _get_tc_data(config, sim.config)
    c = sim.circuit
    dfs = []
    for name, data in tqdm(tc_spikes.items(), desc="Iterating over patterns"):
        syn_idx = utils.get_syn_idx(data["edgef_name"], np.unique(data["spiking_gids"]), assembly.gids)
        print(len(syn_idx))
        syn_df = utils.get_proj_properties(c, data["proj_name"], syn_idx,
                                           [Synapse.POST_GID, Synapse.G_SYNX, Synapse.POST_NEURITE_DISTANCE])
        nrn_df = c.cells.get(syn_df[Synapse.POST_GID], [Cell.LAYER, Cell.MTYPE])
        data = np.concatenate((syn_df[[Synapse.G_SYNX, Synapse.POST_NEURITE_DISTANCE]].to_numpy(),
                               nrn_df.loc[syn_df[Synapse.POST_GID]].to_numpy()), axis=1)
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

