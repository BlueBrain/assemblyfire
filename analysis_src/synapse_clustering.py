"""
...
last modified: András Ecker 01.2022
"""

import os
from tqdm import tqdm
import numpy as np
from scipy.stats import poisson
from scipy.spatial.distance import pdist, squareform

import assemblyfire.utils as utils
from assemblyfire.config import Config


def syn_distances(syn_df, mask_col, xzy_cols):
    """Return (Euclidean) distance between synapses on the same section (the rest is masked with nans)"""
    dists = squareform(pdist(syn_df[xzy_cols].to_numpy()))
    mask = squareform(pdist(syn_df[mask_col].to_numpy().reshape(-1, 1)))
    dists[mask > 0] = np.nan
    np.fill_diagonal(dists, np.nan)
    return dists


def distance_number_model(dists, assembly_frac, target_range=10.0, fig_name=None):
    """Creates a cummulative histogram of (valid) inter-synapse distances, fits a line to it
    and based on the slope returns and underlying Poisson model (the mathematical assumption behind the Poisson is
    that the distribution should be uniform (histogram should be flat) and the cumulative a straight line)
    use fig_name != None to save a figure and visually verify"""
    dists = squareform(dists, checks=False)  # convert back to condensed form to not count every distance twice
    dist_samples = dists[~np.isnan(dists)]
    d_bins = np.arange(2.0 * target_range)  # *2 is totally arbitrary... it's just bigger than the target range
    hist, _ = np.histogram(dist_samples, d_bins)
    cum = np.cumsum(hist) / dists.shape[0]
    fit = np.polyfit(d_bins[1:], cum, 1)
    slope = fit[0]
    if fig_name is not None:
        from assemblyfire.plots import plot_synapse_distance_dist
        plot_synapse_distance_dist(d_bins, hist, cum, fit, fig_name)
    return {"assembly": poisson(target_range * slope * assembly_frac),
            "non_assembly": poisson(target_range * slope * (1 - assembly_frac))}


def assembly_synapse_clustering(config_path, debug=False):
    """Loads in assemblies and..."""

    config = Config(config_path)
    sim_paths = utils.get_sim_path(config.root_path)
    c = utils.get_bluepy_circuit(sim_paths.iloc[0])
    gids = utils.get_E_gids(c, "hex_O1")
    # putting this here in order to get an ImportError from `utils.get_bluepy_circuit()` if bluepy is not installed...
    from bluepy.enums import Synapse
    xyz = [Synapse.POST_X_CENTER, Synapse.POST_Y_CENTER, Synapse.POST_Z_CENTER]
    syn_properties = [Synapse.PRE_GID, Synapse.POST_GID, Synapse.POST_SECTION_ID] + xyz

    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)
    #for seed, assembly_grp in assembly_grp_dict.items():
    #    for assembly in tqdm(assembly_grp.assemblies, desc="seed%i syn. clusters" % seed):
    assembly = assembly_grp_dict["seed100"].assemblies[1]  # get one random assembly for testing
    if debug:
        fig_dir = os.path.join(config.fig_path, "seed%i_dists" % seed)
        utils.ensure_dir(fig_dir)
    post_gids = np.random.choice(assembly.gids, 10, replace=False)  # get random subset for testing (should be max simpl.)
    syn_df = utils.get_syn_properties(c, utils.get_syn_idx(c, gids, post_gids), syn_properties)
    for gid in post_gids:
        syn_df_gid = syn_df.loc[syn_df[Synapse.POST_GID] == gid]
        dists = syn_distances(syn_df_gid, Synapse.POST_SECTION_ID, xyz)
        assembly_frac = len(syn_df_gid.loc[syn_df_gid[Synapse.PRE_GID].isin(assembly.gids)]) / len(syn_df_gid)
        if debug:
            fig_name = os.path.join(fig_dir, "assembly%i_a%i_synapse_dists.png" % (assembly.idx[0], gid))
            model = distance_number_model(dists, assembly_frac, fig_name=fig_name)
        else:
            model = distance_number_model(dists, assembly_frac)


if __name__ == "__main__":
    config_path = "../configs/v7_10seeds.yaml"
    assembly_synapse_clustering(config_path)
