"""
Main run function for getting (and saving) connectivity matrix
last modified: András Ecker 09.2022
"""

import logging
from conntility.connectivity import ConnectivityMatrix

from assemblyfire.config import Config
from assemblyfire.utils import get_sim_path, get_bluepy_circuit

L = logging.getLogger("assemblyfire")


def run(config_path):
    """
    Loads in project related info from yaml config file, gets connectivity matrix and saves it to h5 file
    :param config_path: str - path to project config file
    """
    config = Config(config_path)
    L.info("Loading in connectivity matrix and saving it to: %s" % config.h5f_name)
    c = get_bluepy_circuit(get_sim_path(config.root_path).iloc[0])
    load_cfg = {"loading": {"base_target": config.target, "properties": ["x", "y", "z", "mtype"],
                            "atlas": [{"data": "[PH]y", "properties": ["[PH]y"]}]}}
    connectivity_matrix = ConnectivityMatrix.from_bluepy(c, load_cfg)
    connectivity_matrix.to_h5(config.h5f_name, prefix=config.h5_prefix_connectivity, group_name="full_matrix")

