"""
Main run function for getting (and saving) connectivity matrix
last modified: András Ecker 09.2022
"""

import logging
from conntility.connectivity import ConnectivityMatrix

from assemblyfire.config import Config
from assemblyfire.utils import get_bluepy_circuit_from_root_path

L = logging.getLogger("assemblyfire")


def run(config_path):
    """
    Loads in project related info from yaml config file, gets connectivity matrix and saves it to h5 file
    :param config_path: str - path to project config file
    """
    config = Config(config_path)
    L.info("Loading in connectivity matrix and saving it to: %s" % config.h5f_name)
    c = get_bluepy_circuit_from_root_path(config.root_path)
    load_cfg = {"loading": {"base_target": config.target, "properties": ["layer", "x", "y", "z", "mtype",
                                                                         "ss_flat_x", "ss_flat_y", "depth"]}}
    conn_mat = ConnectivityMatrix.from_bluepy(c, load_cfg, load_full=True,
                                              connectome="S1nonbarrel_neurons__S1nonbarrel_neurons__chemical")
    conn_mat.to_h5(config.h5f_name, prefix=config.h5_prefix_connectivity, group_name="full_matrix")

