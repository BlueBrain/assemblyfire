"""
...
last modified: András Ecker 01.2022
"""

# import os
# from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import pdist, squareform

import assemblyfire.utils as utils
from assemblyfire.config import Config


def syn_distances(syn_df, mask_col, xzy_cols):
    """Return (Euclidean) distance between synapses on the same section (the rest is masked with nans)"""
    dists = squareform(pdist(syn_df[xzy_cols].to_numpy()))
    mask = squareform(pdist(syn_df[mask_col].to_numpy().reshape(-1, 1)))
    dists[mask > 0] = np.nan
    # np.fill_diagonal(dists, np.nan)
    return dists


def assembly_synapse_clustering(config_path):
    """Loads in assemblies and..."""

    config = Config(config_path)
    sim_paths = utils.get_sim_path(config.root_path)
    c = utils.get_bluepy_circuit(sim_paths.iloc[0])
    # putting this here in order to get an ImportError from `utils.get_bluepy_circuit()` if bluepy is not installed...
    from bluepy.enums import Synapse
    xyz = [Synapse.POST_X_CENTER, Synapse.POST_Y_CENTER, Synapse.POST_Z_CENTER]
    syn_properties = [Synapse.POST_GID, Synapse.POST_SECTION_ID] + xyz

    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)
    #for seed, assembly_grp in assembly_grp_dict.items():
    #    for assembly in tqdm(assembly_grp.assemblies, desc="%s syn. clusters" % seed):
    #        assembly.idx:
    assembly = assembly_grp_dict["seed100"].assemblies[1]  # get one random assembly for testing
    pre_gids = assembly.gids
    post_gids = np.random.choice(pre_gids, 100, replace=False)  # get random subset for testing (should be max simpl.)
    syn_df = utils.get_syn_properties(c, utils.get_syn_idx(c, pre_gids, post_gids), syn_properties)
    for gid in post_gids:
        syn_df_gid = syn_df.loc[syn_df[Synapse.POST_GID] == gid]
        dists = syn_distances(syn_df_gid, bluepy.Synapse.POST_SECTION_ID, xyz)


if __name__ == "__main__":
    config_path = "../configs/v7_10seeds.yaml"
    assembly_synapse_clustering(config_path)
