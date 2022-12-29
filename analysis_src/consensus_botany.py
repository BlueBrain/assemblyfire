"""
Consensus assembly botany
last modified: András Ecker 12.2022
"""

import os
import numpy as np

from assemblyfire.config import Config
from assemblyfire.topology import AssemblyTopology, simplex_counts_consensus_instantiations
from assemblyfire.utils import load_consensus_assemblies_from_h5
from assemblyfire.plots import plot_assemblies, plot_consensus_mtypes, plot_simplex_counts_consensus


def consensus_botany(config_path):
    """Loads in consensus assemblies and plots cores' location, unions' and cores' mtype composition,
    and simplex counts of instantiations and their consensus' core's"""

    config = Config(config_path)
    consensus_assemblies = load_consensus_assemblies_from_h5(config.h5f_name,
                                                             prefix=config.h5_prefix_consensus_assemblies)
    conn_mat = AssemblyTopology.from_h5(config.h5f_name, prefix=config.h5_prefix_connectivity)

    nrn_df = conn_mat.vertices
    nrn_df = nrn_df.set_index("gid").drop(columns=["x", "y", "z"])
    gids = nrn_df.index.to_numpy()
    core_gids = [assembly.gids for _, assembly in consensus_assemblies.items()]
    # convert to boolean representation in order to use existing plotting functions
    bool_gids = np.hstack([np.in1d(gids, consensus_assemblies["cluster%i" % i].gids, assume_unique=True).reshape(-1, 1)
                           for i in range(len(consensus_assemblies))])
    fig_name = os.path.join(config.fig_path, "consensus_assemblies.png")
    plot_assemblies(bool_gids, np.arange(bool_gids.shape[1]), gids, nrn_df.drop(columns=["mtype"]), fig_name)

    mtypes = nrn_df["mtype"].to_numpy()
    core_mtypes = [mtypes[np.searchsorted(gids, gids_)] for gids_ in core_gids]
    union_gids = [assembly.union.gids for _, assembly in consensus_assemblies.items()]
    union_mtypes = [mtypes[np.searchsorted(gids, gids_)] for gids_ in union_gids]
    plot_consensus_mtypes(mtypes, core_mtypes, union_mtypes, os.path.join(config.fig_path, "consensus_mtypes.png"))

    simplex_counts, simplex_counts_control = simplex_counts_consensus_instantiations(consensus_assemblies, conn_mat)
    fig_name = os.path.join(config.fig_path, "simplex_counts_consensus.png")
    plot_simplex_counts_consensus(simplex_counts, simplex_counts_control, fig_name)


if __name__ == "__main__":
    consensus_botany("../configs/v7_10seeds_np.yaml")
