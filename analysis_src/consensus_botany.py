"""
Consensus assembly botany
last modified: András Ecker 02.2023
"""

import os
import numpy as np

from assemblyfire.config import Config
from assemblyfire.topology import AssemblyTopology, simplex_counts_consensus_instantiations
from assemblyfire.utils import load_consensus_assemblies_from_h5, consensus_dict2assembly_grp
from assemblyfire.plots import plot_assemblies, plot_consensus_mtypes, plot_simplex_counts_consensus


def consensus_botany(config_path):
    """Loads in consensus assemblies and plots cores' location, unions' and cores' mtype composition,
    and simplex counts of instantiations and their consensus' core's"""

    config = Config(config_path)
    assembly_grp = consensus_dict2assembly_grp(load_consensus_assemblies_from_h5(config.h5f_name))
    conn_mat = AssemblyTopology.from_h5(config.h5f_name, prefix=config.h5_prefix_connectivity)

    nrn_df = conn_mat.vertices
    gids, mtypes = nrn_df["gid"].to_numpy(), nrn_df["mtype"].to_numpy()
    core_idx = [assembly.idx[0] for assembly in assembly_grp.assemblies]  # just to work with the plotting fn.
    fig_name = os.path.join(config.fig_path, "consensus_assemblies.png")
    plot_assemblies(assembly_grp.as_bool(), core_idx, assembly_grp.all, nrn_df.set_index("gid"), fig_name)

    core_mtypes = [mtypes[np.in1d(gids, assembly.gids, assume_unique=True)] for assembly in assembly_grp.assemblies]
    union_mtypes = [mtypes[np.in1d(gids, assembly.union.gids, assume_unique=True)] for assembly in assembly_grp.assemblies]
    plot_consensus_mtypes(mtypes, core_mtypes, union_mtypes, os.path.join(config.fig_path, "consensus_mtypes.png"))

    simplex_counts, simplex_counts_control = simplex_counts_consensus_instantiations(assembly_grp, conn_mat)
    fig_name = os.path.join(config.fig_path, "simplex_counts_consensus.png")
    plot_simplex_counts_consensus(simplex_counts, simplex_counts_control, fig_name)


if __name__ == "__main__":
    consensus_botany("../configs/v7_plastic_chunked.yaml")
